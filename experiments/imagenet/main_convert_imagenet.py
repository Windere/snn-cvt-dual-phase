"""
-*- coding: utf-8 -*-

@Time    : 2022/5/14 1:01

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : main_convert_imagenet.py
"""
import sys

sys.path.append('.')
sys.path.append("..")
sys.path.append("../..")
from models.resnet import ResNetX
from models.vgg import rVGG

from models.snn import IsomorphicSNN, SNN
from models.resnet import resnet_specials
from models.IF_neuron import replaceIF, eval_snn

specials = resnet_specials
# todo: sys.path.append 仅适用于测试, 实际调用应在项目根路径下将测试代码作为模块调用
import copy
import os
import numpy as np
import re
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from enum import Enum
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from util.util import Logger
import torchvision.models as models
from model.snn import LIFCell
import torch.nn.functional as F
from util.hooks import RecordHook, SumHook, DPSumHook, DPRecordHook
from model.surrogate_act import NoisyQC, TClamp, NoisySpike, NTClamp, MyFloor
from util.util import transfer_model
from util.transform import RateCoding
import csv
from util.image_augment import CIFAR10Policy, Cutout
from model.surrogate_act import SurrogateHeaviside, RectSurrogate
from util.fold_bn import search_fold_and_remove_bn, StraightThrough, search_fold_and_reset_bn

parser = argparse.ArgumentParser(description='Converting ANN into SNN on ImageNet')
parser.add_argument('data', metavar='DIR', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    # choices=model_names,
                    help='model architecture: ' +
                         # ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='./checkpoint/imagenet_resnet34_act_tclamp_wd_0.0001_lr_0.1_cas_0-model_best.pth.tar'
                    # default='./checkpoint/imagenet_resnet34_act_nqc_wd_0.0001_lr_0.001_t_16_cas_1-model_best.pth.tar'
                    , type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--act', metavar='activation function', default='tclamp')
parser.add_argument('--T', default=None, type=int,
                    help='number of time steps.')
parser.add_argument('--ckpt_path', metavar='checkpoint', default='./snn_checkpoint')
parser.add_argument('--log_path', metavar='log dir', default='./log')
parser.add_argument('-d', '--resume_discard', dest='resume_discard', action='store_true',
                    help='discard the params in optimizer and scheduler')
parser.add_argument('--aug', action='store_true',
                    help='use color jitter for augmentation')
parser.add_argument('--smooth', action='store_true',
                    help='use label smoothing')
parser.add_argument('--scheduler', default='step', metavar='choose the scheduler', choices=['cos', 'step'])
parser.add_argument('--grid_search', default='./cvt_summary.csv', type=str)
parser.add_argument('--num_epoch2', type=int, default=100,
                    help='epoch for weight calibration')
parser.add_argument('--lr2', type=float, default=3e-4,
                    help='epoch for weight calibration')
parser.add_argument('--patience', type=int, default=10,
                    help='epoch for weight calibration')
parser.add_argument('--cal_train', type=int, default=256,
                    help='epoch for weight calibration')
parser.add_argument('--cal_val', type=int, default=128,
                    help='epoch for weight calibration')
parser.add_argument('--mem_cal_bz', type=int, default=1024,
                    help='epoch for weight calibration')
parser.add_argument('--cal_bz', type=int, default=64,
                    help='epoch for weight calibration')
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


def getModelName(args):
    model_name = 'ann_' + args.arch + '_act_' + args.act + '-' + 't_' + str(
        args.T) + '_epoch2_' + str(args.num_epoch2) + '_lr2_' + str(args.lr2) + '_pat_' + str(
        args.patience) + '_' + str(
        args.cal_train) + '_' + str(args.cal_val)
    model_name += '_cas_' + str(len([n for n in os.listdir(args.ckpt_path) if model_name in n]) + 1)
    return model_name

def extractThreshBFS(qcs_model, reset=False):
    # todo: whether there's a better method to guanrantee the order
    thresh_list = []
    name_list = []
    with torch.no_grad():
        queue = [qcs_model]
        idx = 0
        while len(queue) > 0:
            module = queue.pop()
            for name, child in module.named_children():
                if name == 'act_fun': continue  # skip the template
                if isinstance(child, (NoisyQC, TClamp, NTClamp)):
                    thresh_list.append(child.thresh.detach().item())
                    name_list.append(name)
                    if reset:
                        setattr(child, 'thresh', None)
                    # child.thresh = child.thresh.detach().clone()
                    idx += 1
                else:
                    queue.insert(0, child)
    return name_list, thresh_list


def replaceAct(net, src_act_type, dst_acts, rename_act=True):
    """
    :param net:  the network needed to replace activation_function
    :param src_act_type: the type of source activation functions needed to be replaced with
    :param dst_acts: the target activation need to be replace
                     if isinstance(dst_acts,list), bfs the net and replace src_act with dst_acts[i] one by one
                     else isinstance(dst_acts,nn.Module), replace all the source activation functions with dst_acts
    :return:
    """
    assert isinstance(dst_acts, list) or isinstance(dst_acts,
                                                    nn.Module), 'Destination activation function should be list or nn.Module'
    if rename_act:
        dst_act = dst_acts[0] if isinstance(dst_acts, list) else dst_acts
        setattr(net, 'act_fun', copy.deepcopy(dst_act))
    with torch.no_grad():
        queue = [net]
        idx = 0
        while len(queue) > 0:
            module = queue.pop()
            for name, child in module.named_children():
                if name == 'act_fun': continue  # skip the template
                if isinstance(child, src_act_type):
                    dst_act = (dst_acts[idx] if isinstance(dst_acts, list) else copy.deepcopy(dst_acts))
                    setattr(module, name, dst_act)
                    idx += 1
                else:
                    queue.insert(0, child)


def replaceAvgPool(model):
    with torch.no_grad():
        queue = [model]
        idx = 0
        while len(queue) > 0:
            module = queue.pop()
            for name, child in module.named_children():
                if isinstance(child, nn.MaxPool2d):
                    setattr(module, name,
                            nn.AvgPool2d(kernel_size=child.kernel_size, stride=child.stride, padding=child.padding,
                                         ceil_mode=child.ceil_mode))
                    idx += 1


                else:
                    queue.insert(0, child)


ddp_model = False


def loadModel(args):
    num_perserve = 0
    checkpoint = torch.load(args.resume, map_location='cuda:0')
    best_acc1 = checkpoint['best_acc1']
    if args.act == 'tclamp':
        act_fun = TClamp()
    elif args.act == 'nqc':
        act_fun = NoisyQC(torch.tensor(1.0), args.T, 0.1)
    if args.act == 'ntclamp':
        act_fun = NTClamp()
    src_dict = None
    if ddp_model:
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:10030',
                                world_size=1, rank=0)
        model = models.__dict__[args.arch](pretrained=False)
        replaceAvgPool(model)
        replaceAct(model, nn.ReLU, act_fun, rename_act=False)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])
        src_model = copy.deepcopy(model)
        src_dict = src_model.state_dict()
        print('Load model successfully through pytorch model')
    else:
        src_dict = checkpoint['state_dict']
    # model = models.__dict__[args.arch](pretrained=False)
    if 'res' in args.arch:
        depth = int(re.findall("\d+", args.arch)[0])
        num_perserve = depth
        dst_model = ResNetX(depth, num_class=1000, act_fun=act_fun,
                            modified='modified' in args.arch).cuda()
        dst_model.act_fun = None
    if args.arch == 'vgg16':
        num_perserve = 16
        dst_model = models.__dict__['vgg16_bn']()
        replaceAvgPool(dst_model)
        # replaceAct(model, nn.Dropout, nn.Identity(), rename_act=False)
        # act_fun = TClamp(thresh=10.0)
        # print(act_fun)
        replaceAct(dst_model, nn.Dropout, nn.Identity(), rename_act=False)
        replaceAct(dst_model, nn.ReLU, act_fun, rename_act=False)
        dst_model.act_fun = None
    # print([params.shape for k, params in src_dict.items()])
    # print('*****************************************************')
    # print([params.shape for k, params in dst_model.state_dict().items()])

    # dst_model = rVGG(num_class=1000, dataset='ImageNet',)
    dst_state_dict = transfer_model(src_dict, dst_model.state_dict(), model=False)
    dst_model.load_state_dict(dst_state_dict)
    dst_model.act_fun = act_fun
    return dst_model, best_acc1, num_perserve


def loadModelII(args):
    num_perserve = 0
    checkpoint = torch.load(args.resume)
    best_acc1 = checkpoint['best_acc1']
    if args.act == 'tclamp':
        act_fun = TClamp()
    elif args.act == 'nqc':
        act_fun = NoisyQC(torch.tensor(1.0), args.T, 0.1)
    if args.act == 'ntclamp':
        act_fun = NTClamp()
    src_dict = None
    if ddp_model:
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:10030',
                                world_size=1, rank=0)
        model = models.__dict__[args.arch](pretrained=False)
        replaceAvgPool(model)
        replaceAct(model, nn.ReLU, act_fun, rename_act=False)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])
        src_model = copy.deepcopy(model)
        src_dict = src_model.state_dict()
        print('Load model successfully through pytorch model')
    else:
        src_dict = checkpoint['state_dict']
    # model = models.__dict__[args.arch](pretrained=False)
    if 'res' in args.arch:
        depth = int(re.findall("\d+", args.arch)[0])
        num_perserve = depth
        dst_model = models.__dict__[args.arch]()
        replaceAvgPool(dst_model)
        replaceAct(dst_model, nn.ReLU, act_fun, rename_act=False)

    dst_state_dict = transfer_model(src_dict, dst_model.state_dict(), model=False)
    dst_model.load_state_dict(dst_state_dict)
    # dst_model.act_fun = act_fun
    return dst_model, best_acc1, num_perserve


def loadCIFARModel(args):
    num_perserve = 0
    checkpoint = torch.load(args.resume, map_location='cpu')
    # best_acc = checkpoint['best_acc']
    best_acc = 0
    if args.act == 'tclamp':
        act_fun = TClamp()
    elif args.act == 'nqc':
        act_fun = NoisyQC(torch.tensor(1.0), args.T, 0.1, abs_if=False)  # todo: roll back
    src_dict = None
    if ddp_model:
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:10030',
                                world_size=1, rank=0)
        model = models.__dict__[args.arch](pretrained=False)
        replaceAvgPool(model)
        replaceAct(model, nn.ReLU, act_fun, rename_act=False)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])
        src_model = copy.deepcopy(model)
        src_dict = src_model.state_dict()
        print('Load model successfully through pytorch model')
    else:
        src_dict = checkpoint
    # model = models.__dict__[args.arch](pretrained=False)
    if 'res' in args.arch:
        depth = int(re.findall("\d+", args.arch)[0])
        num_perserve = depth
        dst_model = ResNetX(depth, num_class=10, act_fun=act_fun,
                            modified='modified' in args.arch, old=True).cuda()
        dst_model.act_fun = None
    elif args.arch == 'vgg16':
        dst_model = rVGG(num_class=10, dataset='CIFAR10', kernel_size=3,
                         use_bn=True,
                         act_fun=act_fun, bias=True, hook=False, z_hook=False).cuda()
        dst_model.act_fun = None
        num_perserve = 16
    dst_state_dict = transfer_model(src_dict, dst_model.state_dict(), model=False)
    dst_model.load_state_dict(dst_state_dict)
    dst_model.act_fun = act_fun
    return dst_model, best_acc, num_perserve


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def getImageNet(args):
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    if not args.aug:
        # Data loading code
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    else:
        trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                      ])

        trans = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        train_dataset = datasets.ImageFolder(root=traindir, transform=trans_t)
        val_dataset = datasets.ImageFolder(root=valdir, transform=trans)
    return train_dataset, val_dataset


def loadData(name='CIFAR10', root='/data/wzm/cifar10/', cutout=True, auto_aug=True):
    num_class, normalize, train_data, test_data = None, None, None, None
    train_transform = []
    if name == 'CIFAR10' or name == 'CIFAR100':
        train_transform = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        train_transform.append(CIFAR10Policy())
    train_transform.append(transforms.ToTensor())
    if cutout:
        train_transform.append(Cutout(n_holes=1, length=16))
    if name == 'CIFAR10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        num_class = 10
    elif name == 'CIFAR100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        num_class = 100
    elif name == 'MNIST':
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        num_class = 10
    train_transform.append(normalize)
    train_transform = transforms.Compose(train_transform)
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        normalize
                                        ])
    if name == 'CIFAR100':
        train_data = datasets.CIFAR100(root=root, train=True, download=True,
                                       transform=train_transform)
        val_data = datasets.CIFAR100(root=root, train=False, download=True,
                                     transform=val_transform)
    elif name == 'CIFAR10':
        train_data = datasets.CIFAR10(root=root, train=True, download=True,
                                      transform=train_transform)
        val_data = datasets.CIFAR10(root=root, train=False, download=True,
                                    transform=val_transform)
    elif name == 'MNIST':
        train_data = datasets.MNIST(root=root, train=True, download=True,
                                    transform=train_transform)
        val_data = datasets.MNIST(root=root, train=False, download=True,
                                  transform=val_transform)
    return train_data, val_data, num_class


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg


def extractThresh(root_model, thresh_list=[], name_list=[], reset=False, to_cpu=False):
    # todo: provide the standard bfs oder
    # todo: whether there's a better method to guanrantee the order
    with torch.no_grad():
        for name, child in root_model.named_children():
            if name == 'act_fun': continue  # skip the template
            if isinstance(child, (TClamp, NoisyQC, NTClamp)):
                thresh_list.append(child.thresh.detach().item())
                # if to_cpu:
                #     thresh_list[-1] = thresh_list[-1].cpu()
                name_list.append(name)
                if reset:
                    setattr(child, 'thresh', None)
                # child.thresh = child.thresh.detach().clone()
            else:
                name_list, thresh_list = extractThresh(child, thresh_list, name_list, reset=reset)
    return name_list, thresh_list


def convertSNN(ann, args, decay, thresh, record_interval=None):
    # todo: add a BN checker for alpah=1 & beta=0 or fuse bn & replace it
    ann_ = copy.deepcopy(ann)
    replaceAct(ann_, nn.BatchNorm2d, StraightThrough(), rename_act=False)
    # print(ann)
    snn = IsomorphicSNN(ann_, args.T, decay=decay, thresh=thresh,
                        enable_shift=False, spike_fn=RectSurrogate.apply,
                        mode='psp', enable_init_volt=True, record_interval=record_interval,
                        specials=specials)
    return snn


def runTest(val_ldr, model, evaluator, args=None, encoder=None):
    model.eval()
    device = torch.device('cuda:0')
    with torch.no_grad():
        predict_tot = {}
        label_tot = []
        loss_record = []
        key = 'ann' if encoder is None else 'snn'
        for idx, (ptns, labels) in enumerate(val_ldr):
            # ptns: batch_size x num_channels x T x nNeu ==> batch_size x T x (nNeu*num_channels)
            ptns, labels = ptns.to(device), labels.to(device)
            if encoder is not None:
                ptns = encoder(ptns)
            output = model(ptns)
            if isinstance(output, dict):
                for t in output.keys():
                    if t not in predict_tot.keys():
                        predict_tot[t] = []
                    predict = torch.argmax(output[t], axis=1)
                    predict_tot[t].append(predict)
                loss = evaluator(output[encoder.nb_steps], labels)

            else:
                if key not in predict_tot.keys():
                    predict_tot[key] = []
                loss = evaluator(output, labels)
                # snn.clamp()
                predict = torch.argmax(output, axis=1)
                predict_tot[key].append(predict)
            loss_record.append(loss)
            label_tot.append(labels)

        label_tot = torch.cat(label_tot)
        val_loss = torch.tensor(loss_record).sum() / len(label_tot)
        if 'ann' not in predict_tot.keys() and 'snn' not in predict_tot.keys():
            val_acc = {}
            for t in predict_tot.keys():
                val_acc[t] = torch.mean((torch.cat(predict_tot[t]) == label_tot).float()).item()

        else:
            predict_tot = torch.cat(predict_tot[key])
            val_acc = torch.mean((predict_tot == label_tot).float()).item()
        return val_acc, val_loss


def getActs(model, layer, samples, batch_size=-1, to_cpu=False):
    # todo: strange bug: the results are different when  batch_size = -1 and batch_size = X on gpu, howerver same on cpu
    model.eval()
    layer.eval()
    if batch_size == -1:
        batch_size = len(samples)
    num = len(samples)
    with torch.no_grad():
        index = torch.arange(0, num)
        inputs = []
        extern_inputs = []
        outputs = []
        for i in range(int(np.ceil(num / batch_size))):
            batch_index = index[i * batch_size:min((i + 1) * batch_size, num)]
            batch_samples = samples[batch_index]
            # hooker = SumHook(to_cpu=to_cpu)
            hooker = DPSumHook(to_cpu=to_cpu)
            handler = layer.register_forward_hook(hooker)
            model(batch_samples)
            handler.remove()
            batch_inputs, batch_extern_inputs, batch_outputs = hooker.msync()
            # batch_inputs = hooker.inputs
            # batch_extern_inputs = hooker.extern_inputs
            # batch_outputs = hooker.outputs
            inputs.append(batch_inputs)
            extern_inputs.append(batch_extern_inputs)
            outputs.append(batch_outputs)
        return torch.cat(inputs), torch.cat(extern_inputs), torch.cat(outputs)
        # if batch_size == -1:
        #     if encoder is not None:
        #         samples = encoder(samples)
        #     model(samples)
        #     if encoder is None:
        #         hooker.inputs = hooker.inputs[0]
        #         hooker.outputs = hooker.outputs[0]
        #         hooker.extern_inputs = hooker.extern_inputs[0]
        #     else:
        #         hooker.inputs = torch.stack(hooker.inputs)
        #         hooker.outputs = torch.stack(hooker.outputs)
        #         hooker.extern_inputs = torch.stack(hooker.extern_inputs)
        # else:
        #     num = samples.shape[0]
        #     index = torch.arange(0, num)
        #     inputs, outputs, externs = [], [], []
        #     for i in range(int(math.ceil(num / batch_size))):
        #         batch_index = index[i * batch_size:min((i + 1) * batch_size, num)]
        #         batch_samples = samples[batch_index]
        #         if encoder is not None:
        #             batch_samples = encoder(batch_samples)
        #             model(batch_samples)
        #             batch_inputs, batch_outputs, batch_externs = hooker.reset()
        #             inputs.append(batch_inputs)
        #             outputs.append(batch_outputs)
        #             externs.append(batch_externs)
        #         else:
        #             model(batch_samples)
        #     if encoder is None:
        #         hooker.inputs = torch.cat(hooker.inputs)
        #         hooker.outputs = torch.cat(hooker.outputs)
        #         hooker.extern_inputs = torch.cat(hooker.extern_inputs)
        #     else:
        #         hooker.inputs = torch.cat(inputs, axis=1)
        #         hooker.outputs = torch.cat(outputs, axis=1)
        #         hooker.extern_inputs = torch.cat(externs, axis=1)
    # return hooker


def getFullActs(model, layer, samples, batch_size=128, to_cpu=False):
    # todo: strange bug: the results are different when  batch_size = -1 and batch_size = X on gpu, howerver same on cpu
    model.eval()
    layer.eval()
    if batch_size == -1:
        batch_size = len(samples)
    num = len(samples)
    with torch.no_grad():
        index = torch.arange(0, num)
        inputs = []
        extern_inputs = []
        for i in range(int(np.ceil(num / batch_size))):
            batch_index = index[i * batch_size:min((i + 1) * batch_size, num)]
            batch_samples = samples[batch_index]
            # hooker = SumHook(to_cpu=to_cpu)
            hooker = DPRecordHook(to_cpu=to_cpu)
            handler = layer.register_forward_hook(hooker)
            model(batch_samples)
            handler.remove()
            batch_inputs, batch_extern_inputs = hooker.msync()
            # batch_inputs = hooker.inputs
            # batch_extern_inputs = hooker.extern_inputs
            # batch_outputs = hooker.outputs
            inputs.append(batch_inputs)
            extern_inputs.append(batch_extern_inputs)
            # outputs.append(batch_outputs)
        if batch_extern_inputs.dim() < 3:
            return torch.cat(inputs, axis=1), extern_inputs[0]
        else:
            return torch.cat(inputs, axis=1), torch.cat(extern_inputs, axis=1)
        # if batch_size == -1:
        #     if encoder is not None:
        #         samples = encoder(samples)
        #     model(samples)
        #     if encoder is None:
        #         hooker.inputs = hooker.inputs[0]
        #         hooker.outputs = hooker.outputs[0]
        #         hooker.extern_inputs = hooker.extern_inputs[0]
        #     else:
        #         hooker.inputs = torch.stack(hooker.inputs)
        #         hooker.outputs = torch.stack(hooker.outputs)
        #         hooker.extern_inputs = torch.stack(hooker.extern_inputs)
        # else:
        #     num = samples.shape[0]
        #     index = torch.arange(0, num)
        #     inputs, outputs, externs = [], [], []
        #     for i in range(int(math.ceil(num / batch_size))):
        #         batch_index = index[i * batch_size:min((i + 1) * batch_size, num)]
        #         batch_samples = samples[batch_index]
        #         if encoder is not None:
        #             batch_samples = encoder(batch_samples)
        #             model(batch_samples)
        #             batch_inputs, batch_outputs, batch_externs = hooker.reset()
        #             inputs.append(batch_inputs)
        #             outputs.append(batch_outputs)
        #             externs.append(batch_externs)
        #         else:
        #             model(batch_samples)
        #     if encoder is None:
        #         hooker.inputs = torch.cat(hooker.inputs)
        #         hooker.outputs = torch.cat(hooker.outputs)
        #         hooker.extern_inputs = torch.cat(hooker.extern_inputs)
        #     else:
        #         hooker.inputs = torch.cat(inputs, axis=1)
        #         hooker.outputs = torch.cat(outputs, axis=1)
        #         hooker.extern_inputs = torch.cat(externs, axis=1)
    # return hooker


class EarlyStoping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_state_dict = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        self.best_state_dict = copy.deepcopy(model.state_dict())


def calibInitMemPoten(ann, snn, samples, args, acts_bz=1024, ann_hook_cls=NoisyQC, snn_hook_cls=LIFCell):
    ann.eval()
    snn.eval()
    device = samples.device
    print(device)
    with torch.no_grad():
        snn_cali_layers, ann_cali_layers = [], []
        for n, m in ann.named_modules():
            if n.endswith('act_fun'): continue
            if isinstance(m, ann_hook_cls):
                ann_cali_layers.append(m)
        for m in snn.modules():
            if isinstance(m, snn_hook_cls):
                snn_cali_layers.append(m)
        assert len(snn_cali_layers) == len(
            ann_cali_layers), 'The number of source layers and target layers is mismatching ...'
        snn = nn.DataParallel(snn)
        ann = nn.DataParallel(ann)
        snn = snn.cuda('cuda:0')
        ann = ann.cuda('cuda:0')
        for i in range(len(ann_cali_layers)):
            print('start the potential calibration of layer-{}.'.format(i))
            ann_layer = ann_cali_layers[i]
            snn_layer = snn_cali_layers[i]
            print(i, ann_layer.thresh, snn_layer.thresh)
            assert torch.abs(ann_layer.thresh) == abs(
                snn_layer.thresh), "The threshes in the ann layer and snn layer are mismatching"
            _, _, ann_out = getActs(ann, ann_layer, samples, batch_size=acts_bz, to_cpu=True)
            # spike_samples = encoder(samples)
            _, _, snn_out = getActs(snn, snn_layer, samples, batch_size=acts_bz, to_cpu=True)
            snn_out /= args.T

            # snn_hooker2 = getActs(snn, snn_layer, samples, encoder=encoder, batch_size=-1)
            # print((snn_hooker.outputs != snn_hooker2.outputs).sum())
            # print(snn_layer)
            # print(torch.where(snn_hooker.outputs != snn_hooker2.outputs))
            # torch.cuda.empty_cache()
            # ann_out = ann_hooker.outputs.to(device)
            # snn_out = snn_hooker.outputs.mean(axis=0).to(device)
            # snn_layer.init_mem += ((ann_out.mean(axis=0) - snn_out.mean(
            #     axis=0)) * args.T )
            snn_layer.init_mem += ((ann_out.mean(axis=0) - snn_out.mean(axis=0)) * args.T)
    ann = ann.module
    snn = snn.module
    return ann, snn


def state_forward(snn_module, x, ext_x=0):
    snn_module.reset_membrane_potential()
    out = 0
    # scale = 1
    # if hasattr(snn_module, 'scale'):
    #     scale = snn_module.scale
    for t in range(len(x)):
        out += (snn_module(x[t], ext_x[t]))
    out = out / len(x)
    return out


def kl_loss(batch_output, batch_target):
    batch_size = batch_output.shape[0]
    batch_output = batch_output.view(batch_size, -1)
    batch_target = batch_target.view(batch_size, -1)
    div_loss = nn.KLDivLoss(reduction='batchmean')
    prob_output = F.log_softmax(batch_output, dim=-1)
    prob_target = F.softmax(batch_target, dim=-1)
    return div_loss(prob_output, prob_target)


def caliWeightBPTT(ann, snn, train_samples, val_samples, batch_size=1, ann_hook_cls=NoisyQC, snn_hook_cls=LIFCell,
                   args=None):
    snn_cali_layers, ann_cali_layers = [], []
    for n, m in ann.named_modules():
        # todo: check network arch
        if n.endswith('act_fun'): continue
        if isinstance(m, ann_hook_cls):
            ann_cali_layers.append(m)
    for m in snn.modules():
        if isinstance(m, snn_hook_cls):
            snn_cali_layers.append(m)
    assert len(snn_cali_layers) == len(
        ann_cali_layers), 'The number of source layers and target layers is mismatching ...'
    snn = nn.DataParallel(snn)
    ann = nn.DataParallel(ann)
    snn = snn.cuda('cuda:0')
    ann = ann.cuda('cuda:0')
    num_train = len(train_samples)
    num_val = len(val_samples)
    for i in range(len(ann_cali_layers)):
        snn.eval()
        ann.eval()
        ann_layer, snn_layer = ann_cali_layers[i], snn_cali_layers[i]
        _, _, train_ann_out = getActs(ann, ann_layer, train_samples, args.mem_cal_bz, to_cpu=True)
        train_snn_in, train_snn_extin = getFullActs(snn, snn_layer, train_samples, args.mem_cal_bz, to_cpu=True)
        # torch.cuda.empty_cache()

        _, _, val_ann_out = getActs(ann, ann_layer, val_samples, args.mem_cal_bz, to_cpu=True)
        val_snn_in, val_snn_extin = getFullActs(snn, snn_layer, val_samples, args.mem_cal_bz, to_cpu=True)
        torch.cuda.empty_cache()
        print([n for n, v in snn_layer.named_parameters()])
        # snn_layer.init_mem = nn.Parameter(torch.Tensor(snn_layer.init_mem).cuda(device_ids[0]), requires_grad=True)
        snn_layer.spike_fn = NoisySpike(sig='tanh')
        # snn_layer.thresh = nn.Parameter(torch.tensor([snn_layer.thresh]).cuda(device_ids[0]), requires_grad=True)
        snn_layer = snn_layer.cuda(device_ids[0])
        # if snn_layer.func.bias is None:
        #     snn_layer.func.bias = nn.Parameter(torch.zeros(snn_layer.func.weigth.shape).cuda(device_ids[0]), requires_grad=True)
        # print(type(snn_layer.init_mem))
        # setattr(snn_layer, 'init_mem', nn.Parameter(torch.Tensor(snn_layer.init_mem), requires_grad=True).cuda(device_ids[0]))
        # print(snn_layer)
        # snn_layer.func.weight.requires_grad = False
        # snn_layer.func.bias.requires_grad = False
        print([n for n, p in snn_layer.named_parameters()])
        optimizer = optim.Adam(snn_layer.parameters(), lr=args.lr2, weight_decay=0, amsgrad=True)
        # optimizer = optim.Adam([
        #     {'params': snn_layer.func.bias},
        #     {'params': snn_layer.init_mem},
        #     {'params': snn_layer.spike_fn.sig.alpha}
        # ], lr=3e-4, weight_decay=0,
        #     amsgrad=True)
        # if args.optim_name2 == 'ADAM':
        #     optimizer = optim.Adam(snn_layer.parameters(), lr=args.lr2, weight_decay=args.wd2,
        #                            amsgrad=True)
        #     # optimizer = optim.Adam([{'params': snn_layer.func.parameters()}], lr=args.lr2, weight_decay=args.wd2,
        #     #                        amsgrad=True)
        # elif args.optim_name2 == 'SGDM':
        #     optimizer = optim.SGD(snn_layer.parameters(), lr=args.lr2, weight_decay=args.wd2,
        #                           momentum=0.9)
        #     # optimizer = optim.SGD([{'params': snn_layer.func.parameters()}], lr=args.lr2, weight_decay=args.wd2,
        #     #                       momentum=0.9)
        # elif args.optim_name2 == 'RADAM':
        #     optimizer = RAdam(snn_layer.parameters(), lr=args.lr2, weight_decay=args.wd2)
        #
        #     # optimizer = RAdam([{'params': snn_layer.func.parameters()}], lr=args.lr2, weight_decay=args.wd2)
        # print(args.lr2)
        # optimizer = optim.SGD([snn_layer.func.weight], lr=snn_lr, weight_decay=snn_wd, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch2, eta_min=0.)

        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        # if warmup_step > 0:
        #     scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_step,
        #                                               after_scheduler=scheduler)

        # evaluator = nn.MSELoss()
        # evaluator = nn.SmoothL1Loss()
        # evaluator = cosine_loss
        evaluator = kl_loss
        # evaluator = js_loss

        # evaluator = torch.cosine_similarity
        stoper = EarlyStoping(patience=args.patience, verbose=False)
        print('Start ajusting weight with bptt for {}-th layer'.format(i))
        for iepoch in range(args.num_epoch2):
            # if warmup_step > 0:
            #     scheduler_warmup.step(iepoch)
            index = torch.randperm(num_train)
            train_loss_tot = 0
            snn.train()
            snn_layer.train()
            for ibatch in range(int(np.ceil(num_train / batch_size))):
                # if (ibatch + 1) * batch_size > num_samples: break
                batch_index = index[ibatch * batch_size:min((ibatch + 1) * batch_size, num_train)]
                batch_input = train_snn_in[:, batch_index]
                if train_snn_extin.dim() < 3:
                    batch_extin = train_snn_extin
                else:
                    batch_extin = train_snn_extin[:, batch_index]
                batch_target = train_ann_out[batch_index]
                optimizer.zero_grad()
                batch_input = batch_input.cuda(device=device_ids[0])
                batch_target = batch_target.cuda(device=device_ids[0])
                batch_extin = batch_extin.cuda(device=device_ids[0])
                # batch_output = snn_layer(batch_input).mean(axis=0)
                batch_output = state_forward(snn_layer, batch_input, batch_extin)
                loss = evaluator(batch_output, batch_target)
                # loss = 1 - evaluator(batch_output.view(-1, 1, 1, 1).squeeze(),
                #                      batch_target.view(-1, 1, 1, 1).squeeze(), dim=0)
                train_loss_tot += loss.detach().cpu()
                loss.backward()
                optimizer.step()
                # print(snn_layer.init_mem.mean())
            # if warmup_step == 0:
            #     scheduler.step(train_loss_tot)
            scheduler.step()
            snn.eval()
            snn_layer.eval()
            val_loss_tot = 0
            with torch.no_grad():
                index = torch.arange(0, num_val)
                for ibatch in range(int(np.ceil(num_val / batch_size))):
                    # if (ibatch + 1) * batch_size > num_samples: break
                    batch_index = index[ibatch * batch_size:min((ibatch + 1) * batch_size, num_val)]
                    batch_input = val_snn_in[:, batch_index]
                    if val_snn_extin.dim() < 3:
                        batch_extin = val_snn_extin
                    else:
                        batch_extin = val_snn_extin[:, batch_index]
                    batch_target = val_ann_out[batch_index]
                    batch_input = batch_input.cuda(device=device_ids[0])
                    batch_target = batch_target.cuda(device=device_ids[0])
                    batch_extin = batch_extin.cuda(device=device_ids[0])
                    batch_output = state_forward(snn_layer, batch_input, batch_extin)
                    loss = evaluator(batch_output, batch_target)
                    # loss = 1 - evaluator(batch_output.view(-1, 1, 1, 1).squeeze(),
                    #                      batch_target.view(-1, 1, 1, 1).squeeze(), dim=0)
                    val_loss_tot += loss.cpu()
            if iepoch % 20 == 0:
                print('Epoch {} with train loss {}, validation loss {}'.format(iepoch, train_loss_tot, val_loss_tot))
            stop_if = stoper(val_loss_tot, snn_layer)
            if (stop_if):
                print('Early Stopping at Epoch {}'.format(iepoch))
                break
        snn_layer.load_state_dict(stoper.best_state_dict)
    ann = ann.module
    snn = snn.module
    return ann, snn


def dp_test(model, val_ldr, evaluator, args):
    model = nn.DataParallel(model)
    model = model.cuda('cuda:0')
    acc1 = validate(val_ldr, model, evaluator, args)
    model = model.module
    return model, acc1


def sampleData(data_ldr, num_samples):
    input_samples = []
    target_samples = []
    for (input, target) in data_ldr:
        input_samples.append(input)
        target_samples.append(target)
        if len(input_samples) * len(input_samples[0]) >= num_samples:
            break
    input_samples = torch.cat(input_samples, dim=0)[:num_samples]
    target_samples = torch.cat(target_samples, dim=0)[:num_samples]
    return input_samples, target_samples


# def extractState(snn):
#     for


if __name__ == '__main__':
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True

    noise_prob = 0.1
    args = parser.parse_args()
    train_sampler = None
    device_ids = [0]
    model_name = getModelName(args)
    results = {'model name': model_name}

    log_path = os.path.join(args.log_path, model_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    sys.stdout = Logger(os.path.join(log_path, model_name + '.log'))
    sys.stderr = Logger(os.path.join(log_path, model_name + '.log'))
    ann, acc1, num_perserve = loadModel(args)
    # ann, acc1, num_perserve = loadModelII(args)

    # ann, acc1, num_perserve = loadCIFARModel(args)

    print('The pretrained model obtains the top-1 accuracy {}'.format(acc1))
    # print(model)
    # ann = nn.DataParallel(ann, device_ids=device_ids)
    # ann = ann.cuda(device=device_ids[0])
    # todo: DataParallel
    # todo: Distributed DataParallel
    train_data, val_data = getImageNet(args)
    # train_data, val_data, _ = loadData()
    train_ldr = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_ldr = torch.utils.data.DataLoader(val_data,
                                          batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.workers, pin_memory=True)
    evaluator = nn.CrossEntropyLoss()
    acc1 = 0
    ann, acc1 = dp_test(ann, val_ldr, evaluator, args)
    # acc1 = validate(val_ldr, ann, evaluator, args)
    print('The loaded model obtains the top-1 accuracy {}'.format(acc1))
    search_fold_and_remove_bn(ann, replace=True)
    # ann, acc1 = dp_test(ann, val_ldr, evaluator, args)

    # acc1 = validate(val_ldr, ann, evaluator, args)
    print('ANN after Folding BN into Weights: {}'.format(acc1))
    # ann = ann.module
    if args.act == 'tclamp':
        # name_list, thresh_list = extractThresh(ann, thresh_list=[], to_cpu=True)
        name_list, thresh_list = extractThreshBFS(ann)
        print(thresh_list)
        replaceAct(ann, TClamp,
                   [NoisyQC(torch.abs(torch.tensor(thresh)).cuda(device_ids[0]), nb_steps=args.T, noise_prob=noise_prob,
                            detach=False) for thresh in
                    thresh_list])
    elif args.act == 'ntclamp':
        # name_list, thresh_list = extractThresh(ann, thresh_list=[], to_cpu=True)
        name_list, thresh_list = extractThreshBFS(ann)
        print(thresh_list)
        replaceAct(ann, NTClamp,
                   [NoisyQC(torch.tensor(thresh).cuda(device_ids[0]), nb_steps=args.T, noise_prob=noise_prob,
                            detach=False, abs_if=False) for thresh in
                    thresh_list])
    # ann.act_fun = None
    # torch.save(ann.state_dict(),'./debug_imagenet.pth')
    # ann = nn.DataParallel(ann, device_ids=device_ids)
    # acc1 = validate(val_ldr, ann, evaluator, args)
    ann, acc1 = dp_test(ann, val_ldr, evaluator, args)
    results['qc ann'] = acc1
    print('ANN after Quantitation and Clamping: {}'.format(acc1))
    print(ann)
    # ann = ann.module
    name_list, thresh_list = extractThresh(ann, thresh_list=[], to_cpu=True)
    if args.act == 'tclamp' or args.act == 'nqc':
        thresh_list = [abs(thresh) for thresh in thresh_list]
    thresh_list.append(1)
    print(thresh_list)
    # /data/wzm/imagenet/   --T 16 -j 32 --gpu 0 -p 100 --num_epoch2 100 -b 256 --resume ./checkpoint/imagenet_resnet34_act_nqc_wd_0.0001_lr_0.001_t_16_cas_1-model_best.pth.tar --act nqc
    snn = convertSNN(ann, args, decay=[1 for i in range(num_perserve)], thresh=thresh_list,
                     record_interval=None).cuda(device=device_ids[0])
    # snn = SNN(ann, args.T, NoisyQC)
    # snn = copy.deepcopy(ann)
    # replaceIF(snn, NoisyQC)
    # snn = nn.DataParallel(snn, device_ids=device_ids)
    # snn = snn.cuda(device=device_ids[0])
    # print(eval_snn(val_ldr, snn, torch.device('cuda:0'), sim_len=args.T))

    snn, snn_acc = dp_test(snn, val_ldr, evaluator, args)
    results['raw snn'] = snn_acc
    print('SNN with Shared Weights and Bias: {}'.format(snn_acc))
    best_acc = snn_acc
    best_model = copy.deepcopy(snn)
    samples, _ = sampleData(train_ldr, args.cal_train)  # todo: support incremental data calibration
    # print(args.cal_train)
    samples = samples.cuda(device=device_ids[0])
    val_samples, _ = sampleData(val_ldr, args.cal_val)
    val_samples = val_samples.cuda(device=device_ids[0])
    # ann, snn = calibInitMemPoten(ann, snn, samples, args, args.mem_cal_bz)
    snn, snn_acc = dp_test(snn, val_ldr, evaluator, args)
    if snn_acc > best_acc:
        best_acc = snn_acc
        best_model = copy.deepcopy(snn)
    # snn = nn.DataParallel(snn, device_ids=device_ids)
    # snn = snn.cuda(device=device_ids[0])
    # snn_acc = validate(val_ldr, snn, evaluator, args)
    # print(eval_snn(val_ldr, snn, torch.device('cuda:0'), sim_len=args.T))
    # snn_acc, _ = runTest(val_ldr, snn, nn.CrossEntropyLoss(), encoder=RateCoding(args.T, method=0), args=args)
    results['poten cal'] = snn_acc
    print('SNN after Initial Potential Calibration: {}'.format(snn_acc))
    ann, snn = caliWeightBPTT(ann, snn, samples, val_samples, batch_size=args.cal_bz, args=args)
    snn, snn_acc = dp_test(snn, val_ldr, evaluator, args)
    results['bptt cal'] = snn_acc
    if snn_acc > best_acc:
        best_acc = snn_acc
        best_model = copy.deepcopy(snn)
    print('SNN after Weight Calibration: {}'.format(snn_acc))
    torch.save({
        'state': best_model.state_dict(),
        'thresh': thresh_list,
        'acc1': best_acc,
        'model': best_model,
        'config': args
    }, os.path.join(args.ckpt_path, model_name + '.pth'))
    print('saving model {} with best accuracy {}.'.format(model_name, snn_acc))
    if args.grid_search is not None:
        write_head = not os.path.exists(args.grid_search)
        with open(args.grid_search, 'a+') as f:
            defaults = parser.parse_args([args.data])
            cmd = [' --' + str(k) + ' ' + str(v) for k, v in set(vars(args).items()) - set(vars(defaults).items())]
            results['cmd'] = ' '.join(['python ' + os.path.basename(__file__)] + cmd)
            writer = csv.DictWriter(f, fieldnames=list(results.keys()))
            if write_head:   writer.writeheader()
            writer.writerow(results)
