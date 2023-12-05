"""
-*- coding: utf-8 -*-

@Time    : 2021/12/27 16:05

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : main.py
"""
import sys
import math
from util.hooks import RecordHook

sys.path.append('.')
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../..")
from models.resnet import ResNetX
from models.vgg import rVGG
from util.transform import RateCoding

from models.snn import IsomorphicSNN, SNN
from models.resnet import resnet_specials
from models.IF_neuron import replaceIF, eval_snn
from util.util import setup_seed

import os
import re
import copy
import time
import torch
import argparse
import torch.nn as nn
from enum import Enum
import torchvision.models as models

from model.snn import LIFCell
from model.surrogate_act import NoisyQC, TClamp, NoisySpike, NTClamp, MyFloor
from util.fold_bn import search_fold_and_remove_bn, StraightThrough, search_fold_and_reset_bn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model.surrogate_act import SurrogateHeaviside, RectSurrogate
from config import args


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

def getActs(model, layer, samples, batch_size=-1, encoder=None, to_cpu=False):
    # todo: strange bug: the results are different when  batch_size = -1 and batch_size = X on gpu, howerver same on cpu
    model.eval()
    layer.eval()
    with torch.no_grad():
        hooker = RecordHook(to_cpu=to_cpu)
        handler = layer.register_forward_hook(hooker)
        if batch_size == -1:
            if encoder is not None:
                samples = encoder(samples)
            model(samples)
            if encoder is None:
                hooker.inputs = torch.stack(hooker.inputs)
                hooker.outputs = torch.stack(hooker.outputs)
                hooker.extern_inputs = torch.stack(hooker.extern_inputs)
            else:
                hooker.inputs = torch.stack(hooker.inputs)
                hooker.outputs = torch.stack(hooker.outputs)
                hooker.extern_inputs = torch.stack(hooker.extern_inputs)
        else:
            num = samples.shape[0]
            index = torch.arange(0, num)
            inputs, outputs, externs = [], [], []
            for i in range(int(math.ceil(num / batch_size))):
                batch_index = index[i * batch_size:min((i + 1) * batch_size, num)]
                batch_samples = samples[batch_index]
                if encoder is not None:
                    batch_samples = encoder(batch_samples)
                    model(batch_samples)
                    batch_inputs, batch_outputs, batch_externs = hooker.reset()
                    inputs.append(batch_inputs)
                    outputs.append(batch_outputs)
                    externs.append(batch_externs)
                else:
                    model(batch_samples)
            if encoder is None:
                hooker.inputs = torch.cat(hooker.inputs)
                hooker.outputs = torch.cat(hooker.outputs)
                hooker.extern_inputs = torch.cat(hooker.extern_inputs)
            else:
                hooker.inputs = torch.cat(inputs, axis=1)
                hooker.outputs = torch.cat(outputs, axis=1)
                hooker.extern_inputs = torch.cat(externs, axis=1)
        handler.remove()
    return hooker

def calcSOP(model, samples, batch_size=-1, hook_cls=LIFCell,
            encoder=RateCoding(method=0, nb_steps=16)):
    model.eval()
    device = samples.device
    tot_som_ac = 0
    tot_som_mac = 0
    with torch.no_grad():
        cali_layers = []
        for m in model.modules():
            if isinstance(m, hook_cls):
                cali_layers.append(m)
        for i in range(len(cali_layers)):
            if i == 0: continue
            if i == 1: continue
            if i == len(cali_layers) - 2: continue

            if i == len(cali_layers) - 1: continue

            layer = cali_layers[i]
            hooker = getActs(model, layer, samples, batch_size=args.acts_bz, to_cpu=True,
                             encoder=encoder)
            torch.cuda.empty_cache()
            uniques = torch.unique(hooker.inputs)
            inpts = (hooker.inputs) / uniques.sum()
            uniques = torch.unique(inpts)
            print(uniques)
            assert len(uniques) <= 2, 'the input is not in spikes!'
            spike_counts = inpts.sum(axis=0).to(device)
            analog_counts = torch.ones_like(spike_counts)
            aux_layer = copy.deepcopy(layer)
            aux_layer.weight = torch.nn.Parameter(torch.ones_like(aux_layer.weight))
            aux_layer.bias = torch.nn.Parameter(torch.zeros_like(aux_layer.bias))
            sop_ac = aux_layer(spike_counts)
            sop_mac = aux_layer(analog_counts)
            sop_ac = sop_ac.view(sop_ac.shape[0], -1).sum(axis=1)
            sop_mac = sop_mac.view(sop_mac.shape[0], -1).sum(axis=1)
            tot_som_ac += sop_ac
            tot_som_mac += sop_mac
            # snn_out = snn_hooker.outputs.mean(axis=0).to(device)
            # snn_layer.init_mem += (ann_out.mean(axis=0) - snn_out.mean(
            #     axis=0)) * encoder.nb_steps * ann_layer.thresh
            del hooker
    return tot_som_ac, tot_som_mac


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


def loadModel(args):
    num_perserve = 0
    act_fun = NoisyQC(torch.tensor(1.0), args.T, 0.1)
    src_dict = None
    # model = models.__dict__[args.arch](pretrained=False)
    if 'res' in args.arch:
        depth = int(re.findall("\d+", args.arch)[0])
        num_perserve = depth
        dst_model = ResNetX(depth, num_class=1000, act_fun=act_fun,
                            modified='modified' in args.arch, use_bias=True).cuda()
    elif args.arch == 'vgg16':
        num_perserve = 16
        dst_model = models.__dict__['vgg16_bn']()
        replaceAvgPool(dst_model)
        # replaceAct(model, nn.Dropout, nn.Identity(), rename_act=False)
        # act_fun = TClamp(thresh=10.0)
        # print(act_fun)
        replaceAct(dst_model, nn.Dropout, nn.Identity(), rename_act=False)
        replaceAct(dst_model, nn.ReLU, act_fun, rename_act=False)
        search_fold_and_remove_bn(dst_model, replace=True)
        dst_model.act_fun = copy.deepcopy(act_fun)
        # dst_model.act_fun = None
        # dst_model.act_fun = None
    # dst_state_dict = transfer_model(src_dict, dst_model.state_dict(), model=False)
    # dst_model.load_state_dict(dst_state_dict)
    # dst_model.act_fun = act_fun
    return dst_model


def convertSNN(ann, args, decay, thresh, record_interval=None):
    # todo: add a BN checker for alpah=1 & beta=0 or fuse bn & replace it
    ann_ = copy.deepcopy(ann)
    replaceAct(ann_, nn.BatchNorm2d, StraightThrough(), rename_act=False)
    # print(ann)
    snn = IsomorphicSNN(ann_, args.T, decay=decay, thresh=thresh,
                        enable_shift=False, spike_fn=RectSurrogate.apply,
                        mode='psp', enable_init_volt=True, record_interval=record_interval,
                        specials=resnet_specials)
    snn.spike_fn = None
    return snn


def splitState(state):
    mem_dict = {}
    key_list = [k for k in state if 'init_mem' in k]
    for k in key_list:
        mem_dict[k] = state[k]
        state.pop(k)
    return mem_dict


def setInitMem(snn, mem_dict):
    for n, m in snn.named_modules():
        if isinstance(m, LIFCell):
            if n + '.init_mem' in mem_dict:
                m.init_mem = mem_dict[n + '.init_mem']
            else:
                print('No single init_mem parameters')


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


def testModel():
    state = torch.load(args.resume)
    snn_base = state['model']
    evaluator = nn.CrossEntropyLoss()
    train_data, val_data = getImageNet(args)
    val_ldr = torch.utils.data.DataLoader(val_data,
                                          batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.workers, pin_memory=True)
    print(state['acc1'])

    # print(dp_test(snn_base, val_ldr, evaluator, args))
    # mem_list = detachThresh(snn_base)
    # print(dp_test(snn_base, val_ldr, evaluator, args))
    print(snn_base.state_dict().keys())

    num_perserve = 34
    device_ids = [0]
    ann = loadModel(args)
    snn = convertSNN(ann, args, decay=[1 for i in range(num_perserve)], thresh=state['thresh'],
                     record_interval=None).cuda(device=device_ids[0])
    state = snn_base.state_dict()
    mem_dict = splitState(state)
    snn.load_state_dict(state)
    # print(snn)
    setInitMem(snn, mem_dict)
    print(dp_test(snn, val_ldr, evaluator, args))


if __name__ == '__main__':
    args.dtype = torch.float
    num_workers = 0
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    args.device = device
    globals().update(vars(args))
    setup_seed(args.seed)
    train_data, val_data = getImageNet(args)
    val_ldr = torch.utils.data.DataLoader(val_data,
                                          batch_size=256, shuffle=False,
                                          num_workers=args.workers, pin_memory=True)
    sop_samples, _ = sampleData(val_ldr, 8)  # num of train samples vs. num of val samples = 4 : 1
    torch.save(sop_samples, './sop_samples.data')
    state = torch.load(args.resume, map_location=device)
    snn = state['model']
    # # snn.record_interval = args.record_interval
    print(snn)
    sop_samples = torch.load('./sop_samples.data').to(device)
    hook_cls = (nn.Conv2d, nn.Linear)
    sop_ac, sop_mac = calcSOP(model=snn, samples=sop_samples, hook_cls=hook_cls,
                              encoder=None)
    print('SOP in SNN: {}, SOP in ANN {}'.format(sop_ac / 1e9, sop_mac / 1e9))
    print('SNN Avg: {}, Std {}'.format(torch.mean(sop_ac / 1e9), torch.std(sop_ac / 1e9)))
    print('ANN Avg: {}, Std {}'.format(torch.mean(sop_mac / 1e9), torch.std(sop_mac / 1e9)))

    # print('Nergy in SNN: {}, SOP in ANN {}'.format(sop_ac / 1e9, sop_mac / 1e9))
    print('SNN Energy Avg: {}, Std {}'.format(0.9 * torch.mean(sop_ac / 1e9), 0.9 * torch.std(sop_ac / 1e9)))
    print('ANN Energy Avg: {}, Std {}'.format(4.6 * torch.mean(sop_mac / 1e9), 4.6 * torch.std(sop_mac / 1e9)))
