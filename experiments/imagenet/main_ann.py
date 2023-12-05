"""
-*- coding: utf-8 -*-

@Time    : 2022/4/25 0:07

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : main_ann.py
"""
import argparse
import os
import sys

sys.path.append('.')
sys.path.append("..")
sys.path.append("../..")
from models.resnet import ResNetX

import copy
import re
import random
import shutil
import time
import warnings
from enum import Enum
from tqdm import tqdm
import csv
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from util.util import LabelSmoothing
from torch.utils.tensorboard import SummaryWriter

from util.util import Logger
from model.surrogate_act import Clamp, Quantize, TrainableQCS, TClamp, NoisyQC, NTClamp

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', default='imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--wd_fac', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--optim', metavar='optimizer', default='sgdm')
parser.add_argument('--act', metavar='activation function', default='tclamp')
parser.add_argument('--T', default=None, type=int,
                    help='number of time steps.')
parser.add_argument('--ckpt_path', metavar='checkpoint', default='./checkpoint')
parser.add_argument('--log_path', metavar='log dir', default='./log')
parser.add_argument('-d', '--resume_discard', dest='resume_discard', action='store_true',
                    help='discard the params in optimizer and scheduler')
parser.add_argument('--aug', action='store_true',
                    help='use color jitter for augmentation')
parser.add_argument('--smooth', action='store_true',
                    help='use label smoothing')
parser.add_argument('--split', action='store_true',
                    help='use optimizer settings splittly')
parser.add_argument('--scheduler', default='step', metavar='choose the scheduler', choices=['cos', 'step'])
parser.add_argument('--grid_search', default='./summary.csv', type=str)
best_acc1 = 0


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


def get_model_name(model_name, args):
    model_name = model_name + 'imagenet' + '_' + args.arch.lower() + '_act_' + args.act + '_wd_' + str(
        args.weight_decay) + '_lr_' + str(args.lr)
    if args.act == 'nqc':
        model_name += '_t_' + str(args.T)
    cas_num = len([one for one in os.listdir(args.ckpt_path) if one.startswith(model_name)])
    model_name += '_cas_' + str(cas_num)
    print('model name: ' + model_name)
    return model_name


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


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


def regular_set(model, paras=([], [], [])):
    if model is None:
        return paras
    for n, module in model._modules.items():
        if module is None: continue
        if isinstance(module, (NoisyQC, TClamp, NTClamp)) and hasattr(module, "thresh"):
            for name, para in module.named_parameters():
                paras[0].append(para)
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)

    return paras


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    # replaceAvgPool(model)
    if args.act == 'tclamp':
        act_fun = TClamp()
    elif args.act == 'nqc':
        act_fun = NoisyQC(torch.tensor(1.0), args.T, 0.1)
    elif args.act == 'ntclamp':
        act_fun = NTClamp()
    elif args.act == 'relu':
        act_fun = nn.ReLU()
    # replaceAct(model, nn.ReLU, act_fun, rename_act=False)
    if 'res' in args.arch:
        depth = int(re.findall("\d+", args.arch)[0])
        model = ResNetX(depth, num_class=1000, act_fun=act_fun,
                        modified='modified' in args.arch)
    elif args.arch == 'vgg16':
        model = models.__dict__['vgg16_bn']()
        replaceAvgPool(model)
        # replaceAct(model, nn.Dropout, nn.Identity(), rename_act=False)
        if args.act == 'tclamp':
            act_fun = TClamp(thresh=10.0)
        replaceAct(model, nn.ReLU, act_fun, rename_act=False)
        # model =
        pass
    model.act_fun = None
    model_name = get_model_name('', args)
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        args.log_path = os.path.join(args.log_path, model_name)
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
        writer = SummaryWriter(args.log_path)
        sys.stdout = Logger(os.path.join(args.log_path, model_name + '.log'))
        sys.stderr = Logger(os.path.join(args.log_path, model_name + '.log'))
        print(model)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.smooth:
        criterion = LabelSmoothing().cuda(args.gpu)
    params = model.state_dict()
    if args.split:
        params = regular_set(model)
        params = [
            {'params': params[0], 'weight_decay': args.wd_fac},
            {'params': params[1], 'weight_decay': args.weight_decay},
            {'params': params[2], 'weight_decay': 0}
        ]
    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.optim == 'adam':
        optimizer = torch.optim.AdamW(params, args.lr, weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            print('loading best top-1 acc {}'.format(best_acc1))
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                # best_acc1 = best_acc1.to(args.gpu)
                pass
            if args.act == 'nqc':
                best_acc1 = 0
            model.module.load_state_dict(checkpoint['state_dict'])
            if not args.resume_discard:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    train_dataset, val_dataset = getImageNet(args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        # val_sampler = None
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size * 4, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        print('Resumed Top-1 Acc. {:.2f}:'.format(validate(val_loader, model, criterion, args)))
        return

    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        print('The best accuracy now is {}\n Is this epoch best: {}'.format(best_acc1, is_best))
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': args
            }, is_best, prefix=os.path.join(args.ckpt_path, model_name + '-'))
            writer.add_scalar('Top-1 Acc.', acc1, epoch)
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        if args.grid_search is not None:
            with open(args.grid_search, 'a+') as f:
                writer = csv.DictWriter(f, fieldnames=['model name', 'best acc1', 'config'])
                writer.writerow({'model name': model_name, 'best acc1': best_acc1, 'config': str(vars(args))})


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

    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        # shutil.copyfile(filename, 'model_best.pth.tar')
        print('Changing best... ')
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


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

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count]).to(self.sum.device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
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


if __name__ == '__main__':
    main()
