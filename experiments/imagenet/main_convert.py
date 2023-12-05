"""
-*- coding: utf-8 -*-

@Time    : 2021-11-16 19:37

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : main_convert.py
"""
import sys

sys.path.append('.')
sys.path.append("..")
sys.path.append("../..")
# todo: sys.path.append 仅适用于测试, 实际调用应在项目根路径下将测试代码作为模块调用
import re
import csv
import torch
import copy
import math
import os
import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from util.transform import RateCoding
from util.util import setup_seed, Logger
from util.util import transfer_model, replaceAct
from util.hooks import RecordHook
from torch.utils.tensorboard import SummaryWriter
from models.resnet import ResNetX
import torch.nn.functional as F
from util.fold_bn import search_fold_and_remove_bn, StraightThrough, search_fold_and_reset_bn
from models.snn import IsomorphicSNN
from util.util import AverageMeter, ProgressMeter, Summary
from models.snn_layer import LIFCell, LIFLayer
from model.surrogate_act import QCS, MovingQCS, TrainableQCS, Clamp, NoisyQCS, NoisyQCSII
from model.surrogate_act import SurrogateHeaviside, RectSurrogate
from models.balance_weight_thresh import getMaxAct, balanceWeight, shiftBias
from util.scheduler import GradualWarmupScheduler
from memory_profiler import profile
from util.optim import RAdam
import warnings

warnings.filterwarnings("ignore")
results = {'model_name': '', 'ann': '', 'fold_ann': '', 'ann2snn': '', 'qc_ann': '', 'fintune_qc_ann': '',
           'fintune_qc_ann2snn': '',
           'pc_snn': '', 'wc_pc_snn': '', 'pc_wc_pc_snn': '', 'cmd': ''
           }


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


def runTrain(epoch, train_ldr, optimizer, model, evaluator, args=None):
    # model.cuda(args.device)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_ldr),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    for idx, (ptns, labels) in enumerate(train_ldr):
        # measure data loading time
        torch.cuda.empty_cache()
        data_time.update(time.time() - end)
        ptns = ptns.cuda(args.device)
        labels = labels.cuda(args.device)
        # if encoder is not None:
        #     ptns = encoder(ptns)
        optimizer.zero_grad()
        output = model(ptns)
        # compute gradient and do SGD step
        loss = evaluator(output, labels)
        loss.backward()
        optimizer.step()
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        losses.update(loss.item(), labels.size(0))
        top1.update(acc1[0], labels.size(0))
        top5.update(acc5[0], labels.size(0))
        # predict = torch.argmax(output, axis=1)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_interval == 0:
            progress.display(idx)
    return top1.avg, losses.avg


def runTest(val_ldr, model, evaluator, args=None):
    # model.cuda(args.device)
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_ldr),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for idx, (ptns, labels) in enumerate(val_ldr):
            # ptns: batch_size x num_channels x T x nNeu ==> batch_size x T x (nNeu*num_channels)
            ptns = ptns.to(device)
            labels = labels.to(device)
            # if encoder is not None:
            #     ptns = encoder(ptns)
            # compute output
            output = model(ptns)
            loss = evaluator(output, labels)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), ptns.size(0))
            top1.update(acc1[0], ptns.size(0))
            top5.update(acc5[0], ptns.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_interval == 0:
                progress.display(idx)

        progress.display_summary()

    return top1.avg, losses.avg


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
                if isinstance(child, (TrainableQCS, QCS, NoisyQCS, MovingQCS)):
                    thresh_list.append(child.thresh.detach().item())
                    name_list.append(name)
                    if reset:
                        setattr(child, 'thresh', None)
                    # child.thresh = child.thresh.detach().clone()
                    idx += 1
                else:
                    queue.insert(0, child)
    return name_list, thresh_list


def extractThresh(root_model, thresh_list=[], name_list=[], reset=False):
    # todo: whether there's a better method to guanrantee the order
    with torch.no_grad():
        for name, child in root_model.named_children():
            if name == 'act_fun': continue  # skip the template
            if isinstance(child, (TrainableQCS, QCS, NoisyQCS, MovingQCS, NoisyQCSII)):
                thresh_list.append(child.thresh.detach().item())
                name_list.append(name)
                if reset:
                    setattr(child, 'thresh', None)
                # child.thresh = child.thresh.detach().clone()
            else:
                name_list, thresh_list = extractThresh(child, thresh_list, name_list, reset=reset)
    return name_list, thresh_list


def get_model_name(args):
    model_name = 'imagenet' + '_ann_' + 'resnet34' + '_act_' + args.act_fun_name + '-opt_' + args.optim_name1.lower() + '_wd_' + str(
        args.wd1)
    cas_num = len([one for one in os.listdir(args.log_path) if one.startswith(model_name)])
    model_name += '_cas_' + str(cas_num)
    print('model name: ' + model_name)
    return model_name


def getDataLdr():
    train_data, val_data, num_class = loadData(dataset, data_path, cutout=False,
                                               auto_aug=False)  # todo: try other data aug
    train_ldr = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.train_batch_size, shuffle=True,
                                            pin_memory=True, num_workers=num_workers)
    val_ldr = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                                          pin_memory=True,
                                          num_workers=num_workers)
    if dataset == 'MNIST':
        return train_ldr, train_ldr, val_ldr, num_class
    train_data_, _, num_class = loadData(dataset, data_path, cutout=True,
                                         auto_aug=True)  #
    train_ldr_ = torch.utils.data.DataLoader(dataset=train_data_, batch_size=args.train_batch_size, shuffle=True,
                                             pin_memory=True, num_workers=num_workers)

    return train_ldr_, train_ldr, val_ldr, num_class


def loadModel(args):
    global num_perserve
    if os.path.isfile(args.pretrain):
        print("=> loading checkpoint '{}'".format(args.pretrain))
    state = torch.load(args.pretrain, map_location=args.device_name)
    args.start_epoch = state['epoch']
    best_acc1 = state['best_acc1']
    print('Pretrained model best acc with ReLU:', best_acc1)
    ann = ResNetX(depth=34, act_fun=nn.ReLU(), num_class=1000, modified=False)
    # act_fun_args = {}
    # act_fun = None
    # if args.act_fun_name == 'relu':
    #     act_fun = nn.ReLU
    # elif args.act_fun_name == 'clamp':
    #     act_fun = Clamp
    # elif args.act_fun_name == 'qcs':
    #     act_fun = TrainableQCS
    #     act_fun_args['thresh'] = torch.tensor(1.0)
    #     act_fun_args['nb_step'] = args.T
    # elif args.act_fun_name == 'noisy_qcs':
    #     act_fun = NoisyQCS
    #     act_fun_args = {'thresh': torch.tensor(1.0), 'nb_step': args.T, 'shift': 0, 'trainable': True,
    #                     'p': args.denoise_rate, 'detach': detach}
    # if 'VGG' in args.architecture:
    #     ann = rVGG(architecture, num_class, args.dataset, act_fun=act_fun(**act_fun_args), bias=args.bias,
    #                feature_drop=args.feature_drop,
    #                dropout=args.classifier_drop,
    #                use_bn=args.use_bn, hook=False).to(device, dtype)
    # elif 'Res' in args.architecture:
    #     depth = int(re.findall("\d+", args.architecture)[0])
    #     num_perserve = 3 * depth + 3 if depth >= 50 else 2 * depth + 3
    #     ann = ResNetX(depth, num_class=num_class, act_fun=act_fun(**act_fun_args),
    #                   modified='modified' in args.architecture).to(device, dtype)
    #     if not args.use_bn:
    #         search_fold_and_remove_bn(ann)
    # elif 'LeNet' in args.architecture:
    #     ann = LeNet5(act_fun=act_fun(**act_fun_args)).to(device, dtype)
    #     if not args.use_bn:
    #         search_fold_and_remove_bn(ann)
    # load model and test results
    act_fun = nn.ReLU
    try:
        ann.load_state_dict(state['state_dict'])
    except RuntimeError:
        print('The saved structure and running net structure are mismatching, try sequential weights matching ... ')
        transfer_dict = transfer_model(state['state_dict'], ann.state_dict(), model=False)
        ann.load_state_dict(transfer_dict, strict=False)
    return ann, act_fun


def convertSNN(ann, args, decay, thresh, record_interval=None):
    # todo: add a BN checker for alpah=1 & beta=0 or fuse bn & replace it
    ann_ = copy.deepcopy(ann)
    replaceAct(ann_, nn.BatchNorm2d, StraightThrough(), rename_act=False)
    snn = IsomorphicSNN(ann_, args.T, decay=decay, thresh=thresh,
                        enable_shift=enable_shift, spike_fn=RectSurrogate.apply,
                        mode='spike', enable_init_volt=not enable_shift, record_interval=record_interval,
                        specials=specials)
    return snn


def loadImageNet(root):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    num_class = 1000
    train_data = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_data = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    train_ldr = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.train_batch_size, shuffle=True,
                                            pin_memory=True, num_workers=num_workers)
    val_ldr = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                                          pin_memory=True,
                                          num_workers=num_workers)
    return train_ldr, val_ldr, num_class


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


# @profile
def calibInitMemPoten(ann, snn, samples, batch_size=-1, ann_hook_cls=nn.ReLU, snn_hook_cls=LIFLayer):
    ann.eval()
    snn.eval()
    device = samples.device
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
        for i in range(len(ann_cali_layers)):
            ann_layer = ann_cali_layers[i]
            snn_layer = snn_cali_layers[i]
            # print(i, ann_layer.thresh, snn_layer.thresh)
            assert ann_layer.thresh == snn_layer.thresh, "The threshes in the ann layer and snn layer are mismatching"
            ann_hooker = getActs(ann, ann_layer, samples, batch_size=args.acts_bz, to_cpu=True)
            # spike_samples = encoder(samples)
            snn_hooker = getActs(snn, snn_layer, samples, batch_size=args.acts_bz, to_cpu=True)
            # snn_hooker2 = getActs(snn, snn_layer, samples, encoder=encoder, batch_size=-1)
            # print((snn_hooker.outputs != snn_hooker2.outputs).sum())
            # print(snn_layer)
            # print(torch.where(snn_hooker.outputs != snn_hooker2.outputs))
            torch.cuda.empty_cache()
            ann_out = ann_hooker.outputs.to(device)
            snn_out = snn_hooker.outputs.mean(axis=0).to(device)
            snn_layer.init_mem += (ann_out.mean(axis=0) - snn_out.mean(
                axis=0)) * args.T * ann_layer.thresh
            del ann_hooker
            del snn_hooker


def state_forward(snn_module, x, ext_x=0):
    snn_module.reset_membrane_potential()
    out = []
    # scale = 1
    # if hasattr(snn_module, 'scale'):
    #     scale = snn_module.scale
    out = 0
    for t in range(len(x)):
        # out.append(snn_module(x[t], scale * ext_x[t]))
        out += snn_module(x[t], ext_x[t])

    # out = torch.stack(out)
    return out / len(x)


def getActs(model, layer, samples, batch_size=-1, to_cpu=False):
    # todo: strange bug: the results are different when  batch_size = -1 and batch_size = X on gpu, howerver same on cpu
    model.eval()
    layer.eval()
    with torch.no_grad():
        hooker = RecordHook(to_cpu=to_cpu)
        handler = layer.register_forward_hook(hooker)
        if batch_size == -1:
            model(samples)
            if not isinstance(model, IsomorphicSNN):
                hooker.inputs = hooker.inputs[0]
                hooker.outputs = hooker.outputs[0]
                hooker.extern_inputs = hooker.extern_inputs[0]
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
                if isinstance(model, IsomorphicSNN):
                    # batch_samples = encoder(batch_samples)
                    model(batch_samples)
                    batch_inputs, batch_outputs, batch_externs = hooker.reset()
                    inputs.append(batch_inputs)
                    outputs.append(batch_outputs)
                    externs.append(batch_externs)
                else:
                    model(batch_samples)
            if not isinstance(model, IsomorphicSNN):
                hooker.inputs = torch.cat(hooker.inputs)
                hooker.outputs = torch.cat(hooker.outputs)
                hooker.extern_inputs = torch.cat(hooker.extern_inputs)
            else:
                hooker.inputs = torch.cat(inputs, axis=1)
                hooker.outputs = torch.cat(outputs, axis=1)
                hooker.extern_inputs = torch.cat(externs, axis=1)
        handler.remove()
    return hooker


def cosine_loss(batch_output, batch_target):
    return 1 - torch.cosine_similarity(batch_output.view(-1, 1, 1, 1).squeeze(),
                                       batch_target.view(-1, 1, 1, 1).squeeze(), dim=0)


def kl_loss(batch_output, batch_target):
    div_loss = nn.KLDivLoss()
    prob_output = F.log_softmax(batch_output, dim=-1)
    prob_target = F.softmax(batch_target, dim=-1)
    return div_loss(prob_output, prob_target)


def testCacheModel(path):
    model = torch.load(path)
    runTest(val_ldr, model['net'], nn.CrossEntropyLoss(), args)


def js_loss(net_1_logits, net_2_logits):
    # todo: check error
    raise NotImplementedError('Still have some strange error')
    net_1_probs = F.softmax(net_1_logits, dim=-1)
    net_2_probs = F.softmax(net_2_logits, dim=-1)
    m = 0.5 * (net_1_probs + net_1_probs)
    loss = 0.0
    loss += F.kl_div(m.log(), net_1_logits, reduction="mean")
    loss += F.kl_div(m.log(), net_2_logits, reduction="mean")
    return 0.5 * loss


def caliWeightBPTT(ann, snn, train_samples, val_samples, batch_size=1, ann_hook_cls=nn.ReLU, snn_hook_cls=LIFLayer,
                   encoder=RateCoding(method=0, nb_steps=16), args=None):
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
    num_train = len(train_samples)
    num_val = len(val_samples)
    for i in range(len(ann_cali_layers)):
        snn.eval()
        ann.eval()
        ann_layer, snn_layer = ann_cali_layers[i], snn_cali_layers[i]
        ann_train_hooker = getActs(ann, ann_layer, train_samples, batch_size=args.acts_bz, to_cpu=True)
        snn_train_hooker = getActs(snn, snn_layer, train_samples, batch_size=args.acts_bz, to_cpu=True)
        train_inputs = snn_train_hooker.inputs
        train_extins = snn_train_hooker.extern_inputs
        train_targets = ann_train_hooker.outputs  # todo: check here
        torch.cuda.empty_cache()

        ann_val_hooker = getActs(ann, ann_layer, val_samples, batch_size=args.acts_bz, to_cpu=True)
        snn_val_hooker = getActs(snn, snn_layer, val_samples, batch_size=args.acts_bz, to_cpu=True)
        val_inputs = snn_val_hooker.inputs
        val_extins = snn_val_hooker.extern_inputs
        val_targets = ann_val_hooker.outputs  # todo: check here
        torch.cuda.empty_cache()
        print([n for n, v in snn_layer.named_parameters()])
        if args.optim_name2 == 'ADAM':
            optimizer = optim.Adam(snn_layer.parameters(), lr=args.lr2, weight_decay=args.wd2,
                                   amsgrad=True)
            # optimizer = optim.Adam([{'params': snn_layer.func.parameters()}], lr=args.lr2, weight_decay=args.wd2,
            #                        amsgrad=True)
        elif args.optim_name2 == 'SGDM':
            optimizer = optim.SGD(snn_layer.parameters(), lr=args.lr2, weight_decay=args.wd2,
                                  momentum=0.9)
            # optimizer = optim.SGD([{'params': snn_layer.func.parameters()}], lr=args.lr2, weight_decay=args.wd2,
            #                       momentum=0.9)
        elif args.optim_name == 'RADAM':
            optimizer = RAdam(snn_layer.parameters(), lr=args.lr2, weight_decay=args.wd2)

            # optimizer = RAdam([{'params': snn_layer.func.parameters()}], lr=args.lr2, weight_decay=args.wd2)
        # print(args.lr2)
        # optimizer = optim.SGD([snn_layer.func.weight], lr=snn_lr, weight_decay=snn_wd, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=patience, eta_min=0.)

        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        if warmup_step > 0:
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_step,
                                                      after_scheduler=scheduler)

        # evaluator = nn.MSELoss()
        # evaluator = nn.SmoothL1Loss()
        # evaluator = cosine_loss
        evaluator = kl_loss
        # evaluator = js_loss

        # evaluator = torch.cosine_similarity
        stoper = EarlyStoping(patience=patience, verbose=False)
        print('Start ajusting weight with bptt for {}-th layer'.format(i))
        for iepoch in range(args.num_epoch2):
            if warmup_step > 0:
                scheduler_warmup.step(iepoch)
            index = torch.randperm(num_train)
            train_loss_tot = 0
            snn.train()
            snn_layer.train()
            for ibatch in range(math.ceil(num_train / batch_size)):
                # if (ibatch + 1) * batch_size > num_samples: break
                batch_index = index[ibatch * batch_size:min((ibatch + 1) * batch_size, num_train)]
                batch_input = train_inputs[:, batch_index]
                batch_extin = train_extins[:, batch_index]
                batch_target = train_targets[batch_index]
                optimizer.zero_grad()
                batch_input = batch_input.to(device, dtype)
                batch_target = batch_target.to(device, dtype)
                batch_extin = batch_extin.to(device, dtype)
                # batch_output = snn_layer(batch_input).mean(axis=0)
                batch_output = state_forward(snn_layer, batch_input, batch_extin)
                loss = evaluator(batch_output, batch_target)
                # loss = 1 - evaluator(batch_output.view(-1, 1, 1, 1).squeeze(),
                #                      batch_target.view(-1, 1, 1, 1).squeeze(), dim=0)
                train_loss_tot += loss.detach().cpu()
                loss.backward()
                optimizer.step()
            if warmup_step == 0:
                scheduler.step(train_loss_tot)
            snn.eval()
            val_loss_tot = 0
            with torch.no_grad():
                index = torch.arange(0, num_val)
                for ibatch in range(math.ceil(num_val / batch_size)):
                    # if (ibatch + 1) * batch_size > num_samples: break
                    batch_index = index[ibatch * batch_size:min((ibatch + 1) * batch_size, num_val)]
                    batch_input = val_inputs[:, batch_index]
                    batch_extin = val_extins[:, batch_index]
                    batch_target = val_targets[batch_index]
                    batch_input = batch_input.to(device, dtype)
                    batch_target = batch_target.to(device, dtype)
                    batch_extin = batch_extin.to(device, dtype)
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
        #  todo: clear those hooker by function namespace
        del ann_val_hooker
        del ann_train_hooker
        del snn_train_hooker
        del snn_val_hooker


def runPhaseII(ann, snn, train_ldr, val_ldr, args):
    best_net = None
    samples, _ = sampleData(train_ldr, args.num_samples)
    samples = samples.to(args.device, args.dtype)
    val_samples, _ = sampleData(val_ldr, args.num_samples // 4)  # num of train samples vs. num of val samples = 4 : 1
    val_samples = val_samples.to(args.device, args.dtype)
    calibInitMemPoten(ann, snn, samples, ann_hook_cls=type(ann.act_fun), snn_hook_cls=LIFCell,
                      )
    snn = nn.DataParallel(snn)
    torch.cuda.empty_cache()
    snn_acc1, _ = runTest(val_ldr, snn, nn.CrossEntropyLoss(), args=args)
    snn = snn.module
    best_acc, best_net = snn_acc1, copy.deepcopy(snn)
    print('SNN with Average Membrane Potential Calibration:',
          snn_acc1)
    # return 0
    torch.cuda.empty_cache()
    caliWeightBPTT(ann, snn, samples, val_samples, batch_size=args.bz2, ann_hook_cls=type(ann.act_fun),
                   snn_hook_cls=LIFCell, args=args)
    torch.save({'net': snn}, './snn_cache.pth')
    snn = nn.DataParallel(snn)
    snn_acc2, _ = runTest(val_ldr, snn, nn.CrossEntropyLoss(), args=args)
    snn = snn.module
    print('SNN with BPTT Calibration:',
          snn_acc2)
    if snn_acc2 > best_acc:
        best_acc, best_net = snn_acc2, copy.deepcopy(snn)
    calibInitMemPoten(ann, snn, samples, ann_hook_cls=type(ann.act_fun), snn_hook_cls=LIFCell,
                      )
    snn = nn.DataParallel(snn)
    torch.cuda.empty_cache()
    snn_acc3, _ = runTest(val_ldr, snn, nn.CrossEntropyLoss(), args=args)
    snn = snn.module
    print('SNN with second Average Membrane Potential Calibration:',
          snn_acc3)
    if snn_acc3 > best_acc:
        best_acc, best_net = snn_acc3, copy.deepcopy(snn)
    return (snn_acc1, snn_acc2, snn_acc3, best_acc), best_net


def finetune(model, num_epoch, train_ldr, val_ldr, best_acc, save_if=False, parallel=True):
    # initial varaibles both for record and visulization
    best_dict = model.state_dict()
    start_epoch = 0
    best_epoch = 0
    train_trace, val_trace = dict(), dict()
    train_trace['acc'], train_trace['loss'] = [], []
    val_trace['acc'], val_trace['loss'] = [], []
    writer = SummaryWriter(log_path)
    if parallel:
        model = nn.DataParallel()
    optimizer = optim.SGD(model.parameters(), lr=lr1, momentum=momentum, weight_decay=wd1)
    evaluator = torch.nn.CrossEntropyLoss()
    if scheduler_name == 'COSINE':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epoch)
    elif scheduler_name == 'STEP_LR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_interval, gamma=lr_scale)
    for epoch in tqdm(range(start_epoch, start_epoch + num_epoch)):
        train_acc, train_loss = runTrain(epoch, train_ldr, optimizer, model, evaluator, args=args)
        scheduler.step()
        val_acc, val_loss = runTest(val_ldr, model, evaluator, args=args)
        # saving checkpoint
        print('validation record:', val_loss, val_acc)
        if (val_acc > best_acc):
            with torch.no_grad():
                if parallel:
                    best_dict = copy.deepcopy(model.module.state_dict())
                else:
                    best_dict = copy.deepcopy(model.state_dict())
            best_acc = val_acc
            best_epoch = epoch
            print('Saving model..  with acc {0} in the epoch {1}'.format(best_acc, epoch))
            state = {
                'best_acc': best_acc,
                'best_epoch': epoch,
                'best_net': best_dict,
                'best_net_with_class': model.module if parallel else model,
                'traces': {'train': train_trace, 'val': val_trace},
                'config': args
            }
            if save_if:
                torch.save(state, os.path.join(ckpt_path, model_name + '.pth'))
        # record and log
        train_trace['acc'].append(train_acc)
        train_trace['loss'].append(train_loss)
        val_trace['acc'].append(val_acc)
        val_trace['loss'].append(val_loss)
        # record in tensorboard
        writer.add_scalars('loss', {'val': val_loss, 'train': train_loss},
                           epoch + 1)
        writer.add_scalars('acc', {'val': val_acc, 'train': train_acc},
                           epoch + 1)
        print('Epoch %d: train acc %.5f, test acc %.5f ' % (epoch, train_acc, val_acc))
    print('Finish training: the best traning accuray is {} in epoch {}. \n The relate checkpoint path: {}'.format(
        best_acc,
        best_epoch,
        os.path.join(
            ckpt_path,
            model_name + '.pth')))
    return best_dict, best_acc


def runPhaseI(ann, args, train_ldr, val_ldr, best_acc):
    global model_name
    replaceAct(ann, StraightThrough, nn.BatchNorm2d(), rename_act=False)
    best_dict, ann_acc = finetune(ann, args.num_epoch1, train_ldr, val_ldr, best_acc, save_if=(args.num_epoch1 > 0))
    ann.load_state_dict(best_dict)

    search_fold_and_remove_bn(ann, replace=True)

    # search_fold_and_remove_bn(ann, replace=True)
    # name_list, thresh_list = extractThresh(ann, thresh_list=[], reset=False)
    # thresh_list.append(1)
    # print(thresh_list)
    # snn = convertSNN(ann, thresh=thresh_list, record_interval=True, args=args, decay=torch.ones(num_perserve))
    # snn = convertSNN(ann, thresh=thresh_list, record_interval=True)
    # snn = IsomorphicSNN(copy.copy(ann), T, decay=torch.ones(num_perserve), thresh=torch.ones(num_perserve),
    #                     enable_shift=True,
    #                     mode='spike')
    # snn = SpikingVGG(architecture, num_class=num_class, dataset=args.dataset, dropout=0, feature_drop=0, bias=args.bias,
    #                  readout='mean_cum', neu_mode='spike', decay=torch.ones(num_perserve),
    #                  thresh=torch.tensor(thresh_list)).to(device, dtype)
    # transfer_dict = transfer_model(ann.state_dict(), snn.state_dict(), model=False)
    # snn.load_state_dict(transfer_dict, strict=False)
    snn_acc, _ = runTest(val_ldr, snn, nn.CrossEntropyLoss(), encoder=RateCoding(T, method=0), args=args)
    return (ann, ann_acc), (snn, snn_acc)


if __name__ == '__main__':
    from config.phase_II_cfg import *
    from models.resnet import resnet_specials

    print('Hyper Parameter Config:')
    for k, v in vars(args).items():
        print(' --' + str(k) + ' ' + str(v))
    args.cmd = ['python ' + os.path.basename(__file__)] + args.cmd
    results['cmd'] = ' '.join(args.cmd)
    specials = resnet_specials
    globals().update(vars(args))
    global device, dtype
    setup_seed(seed)
    num_workers = 8
    num_perserve = 34
    dtype = torch.float
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    start_epoch = 0
    RectSurrogate.alpha = 1  # todo: adjust the width
    num_workers = 0
    num_perserve = 34
    momentum = 0.9
    lr_scale = 0.1
    args.device, args.dtype, args.start_epoch = device, dtype, start_epoch
    model_name = get_model_name(args)
    results['model_name'] = model_name
    if not os.path.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    log_path = os.path.join(args.log_path, model_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    sys.stdout = Logger(os.path.join(log_path, model_name + '.log'))
    sys.stderr = Logger(os.path.join(log_path, model_name + '.log'))
    print('Start running on device {}'.format(device))
    # Initialization
    train_ldr, val_ldr, num_class = loadImageNet(args.data_path)
    evaluator = nn.CrossEntropyLoss()
    testCacheModel('./snn_cache.pth')
    ann, act_fun = loadModel(args)
    print(type(ann))
    ann = ann.to(device)
    search_fold_and_remove_bn(ann, replace=True)
    # results['ann'], _ = runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args)
    # print('Pretrained ANN with {}:'.format(act_fun_name), results['ann'])
    percentile = 0.9998  # 72.422   # empirical config
    name, max_acts = getMaxAct(ann, train_ldr, hook_class=nn.ReLU, device=torch.device('cuda'),
                               dtype=torch.float, num_samples=256, percentile=percentile)
    # name, max_acts = getMaxAct(ann, train_ldr, hook_class=nn.ReLU, device=torch.device('cuda'),
    #                            dtype=torch.float, num_samples=256)
    balanceWeight(ann, name, max_acts)
    # results['ann'], _ = runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args)
    # print('Pretrained ANN with {}:'.format(act_fun_name), results['ann'])
    # replaceAct(model, nn.ReLU, Clamp())
    replaceAct(ann, nn.ReLU,
               [NoisyQCSII(torch.tensor(1.0).cuda(args.device), args.T, p=args.denoise_rate, shift=0, trainable=True,
                           detach=False) for i in range(num_perserve)])
    print(ann)
    results['ann'], _ = runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args)
    print('Pretrained ANN with {}:'.format(act_fun_name), results['ann'])
    # exit()
    # # todo: add supports for replace = False [finish]
    # search_fold_and_remove_bn(ann, replace=True)
    # search_fold_and_reset_bn(ann) # update
    # snn_no_bn = copy.deepcopy(ann)
    # search_fold_and_remove_bn(snn_no_bn, replace=True)
    # results['fold_ann'], _ = runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args)
    # print('ANN after Folding BN into Weights:', results['fold_ann'])
    # if args.act_fun_name == 'relu' or args.act_fun_name == 'clamp':
    #     if args.act_fun_name == 'relu':
    #         args.percentile = 0.995  # empirical config
    #         name, max_acts = getMaxAct(ann, train_ldr, hook_class=nn.ReLU, device=device, dtype=dtype,
    #                                    percentile=args.percentile)
    #         balanceWeight(ann, name, max_acts)
    #         print('ANN after Weight Balancing with percentile factor {}:'.format(args.percentile),
    #               runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args))
    #     replaceAct(ann, act_fun,
    #                [NoisyQCS(torch.tensor(1.0), T, p=denoise_rate, shift=0, trainable=True, detach=detach) for i in
    #                 range(num_perserve)])
    # elif args.act_fun_name == 'noisy_qcs':
    #     pass
    # else:
    #     raise NotImplementedError("not organized into this version")
    # replaceAct(ann, act_fun,
    #            [QCS(torch.tensor(1.0), T,  shift=0) for i in range(num_perserve)])

    # best_acc, _ = runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args)
    # results['qc_ann'] = best_acc
    # print('ANN after Quantitation and Clamping : ', best_acc)
    name_list, thresh_list = extractThresh(ann, thresh_list=[])
    thresh_list.append(1)
    print(thresh_list)
    snn = convertSNN(ann, args, decay=torch.ones(num_perserve), thresh=thresh_list,
                     record_interval=None).to(device)
    snn = nn.DataParallel(snn)
    snn_acc, _ = runTest(val_ldr, snn, nn.CrossEntropyLoss(), args=args)
    print('Initial SNN with Shared Weights and Bias:',
          snn_acc)
    snn = snn.module
    runPhaseII(ann, snn, train_ldr, val_ldr, args)

    # results['ann2snn'] = snn_acc[T]

    # Start Running Phase I

    # lr_interval = [int(ratio * num_epoch1) for ratio in args.lr_interval]
    # # log_path = './log'
    # scheduler_name = 'COSINE'  # todo: adjust here
    # args.log_interval = 100
    # args.num_epoch = args.num_epoch1
    # (ann, results['fintune_qc_ann']), (snn, results['fintune_qc_ann2snn']) = runPhaseI(ann, args, train_ldr_, val_ldr,
    #                                                                                    best_acc)
    # results['fintune_qc_ann2snn'] = results['fintune_qc_ann2snn'][T]
    # print('SNN with Shared Weights and Bias:', results['fintune_qc_ann2snn'])
    # Start Running Phase II
    if args.num_epoch2 == 0:
        print('Phase II is skipped as num_epoch2 ')
        exit(1)
    # args.num_samples = 1024
    # args.wd2 = 0
    # args.lr2 = 0.01
    # args.num_epoch2 = 1000
    # best_acc_rec = 0
    # lr2_rec = 0
    # for args.lr2 in [0.1]:
    # if args.grid_search:
    #     # todo:
    #     best_acc_rec = 0
    #     lr2_rec = 0
    #     best_net = None
    #     for args.lr2 in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
    #         acc, snn = runPhaseII(copy.deepcopy(ann), copy.deepcopy(snn), train_ldr, val_ldr, args)
    #         if best_acc_rec < acc:
    #             best_acc_rec = acc
    #             lr2_rec = args.lr2
    #             best_net = snn
    #     print('Best Result with Gradient Search: {} {}'.format(lr2_rec, best_acc_rec))
    # else:
    (results['pc_snn'], results['wc_pc_snn'], results['pc_wc_pc_snn'], best_acc_rec), best_net = runPhaseII(
        copy.deepcopy(ann), copy.deepcopy(snn), train_ldr, val_ldr, args)
    # lr2_rec = args.lr2
    # print('Result: {} {}'.format(args.lr2, best_acc_rec))
    state = {
        'best_acc': best_acc_rec,
        'best_net': best_net.state_dict(),
        'config': args,
        'abstract': results,
        'thresh': thresh_list
    }
    model_name = model_name.replace('ann', 'snn')
    print('Saving best net in path: ', model_name)
    torch.save(state, os.path.join(ckpt_path, model_name + '.pth'))
    if args.grid_search is not None:
        with open(args.grid_search, 'a+') as f:
            writer = csv.DictWriter(f, fieldnames=list(results.keys()))
            # writer.writeheader()
            writer.writerow(results)
        # with open(args.grid_search, 'a+') as f:
        #     f.write(str(args.lr2) + " " + str(best_acc_rec.item()) + " " + model_name + '\n')  # 加\n换行显示

    # args.num_samples = 1024
    # args.wd2 = 0
    # args.lr2 = 0.1
    # args.num_epoch2 = 1000
    # runPhaseII(ann, snn, train_ldr, val_ldr, args)
    # 0.9063
    # SmothL1
    # change into val_ldr validation 0.9111
    # lr = 1, bz2=128
    # lr =1 .25, 0.9157
    # ***********************************************************************
    # setup_seed(seed)
    # dtype = torch.float
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # start_epoch = 0
    # args.device, args.dtype, args.start_epoch = device, dtype, start_epoch
    # num_workers = 0
    # num_perserve = 16
    # train_ldr_, train_ldr, val_ldr, num_class = getDataLdr()
    # act_fun = Clamp
    # ann = rVGG(architecture, num_class, args.dataset, act_fun=act_fun(), bias=args.bias,
    #            feature_drop=args.feature_drop,
    #            dropout=args.classifier_drop,
    #            use_bn=True, hook=False).to(device, dtype)
    # search_fold_and_remove_bn(ann, replace=True)
    #
    # replaceAct(ann, act_fun,
    #            [NoisyQCS(torch.tensor(1.0), T, p=0.5, shift=0, trainable=False) for i in range(num_perserve)])
    # best_dict = torch.load(
    #     './checkpoint/finetune_T_2-cifar10_cut_aug-ann_vgg16_act_clamp-opt_cosine_bn_bias_wd_0_cas_0.pth')
    # ann.load_state_dict(best_dict['best_net'])
    # thresh_list = []
    # with torch.no_grad():
    #     queue = [ann]
    #     idx = 0
    #     while len(queue) > 0:
    #         module = queue.pop()
    #         for name, child in module.named_children():
    #             if name == 'act_fun': continue  # skip the template
    #             if isinstance(child, (TrainableQCS, QCS, NoisyQCS, MovingQCS)):
    #                 thresh_list.append(child.thresh.detach().clone())
    #                 setattr(child, 'thresh', None)
    #                 # child.thresh = child.thresh.detach().clone()
    #                 idx += 1
    #             else:
    #                 queue.insert(0, child)
    # thresh_list.append(1)
    # print(thresh_list)
    # snn = convertSNN(ann, thresh=thresh_list, record_interval=True, args=args, decay=torch.ones(num_perserve))
    # # snn = IsomorphicSNN(copy.copy(ann), T, decay=torch.ones(num_perserve), thresh=torch.ones(num_perserve),
    # #                     enable_shift=True,
    # #                     mode='spike')
    # # snn = SpikingVGG(architecture, num_class=num_class, dataset=args.dataset, dropout=0, feature_drop=0, bias=args.bias,
    # #                  readout='mean_cum', neu_mode='spike', decay=torch.ones(num_perserve),
    # #                  thresh=torch.tensor(thresh_list)).to(device, dtype)
    # # transfer_dict = transfer_model(ann.state_dict(), snn.state_dict(), model=False)
    # # snn.load_state_dict(transfer_dict, strict=False)
    # print('SNN with Shared Weights and Bias:',
    #       runTest(val_ldr, snn, nn.CrossEntropyLoss(), encoder=RateCoding(T, method=0), args=args))
    # args.num_samples = 1024
    # args.wd2 = 0
    # args.lr2 = 0.1
    # args.num_epoch2 = 1000
    # runPhaseII(ann, snn, train_ldr, val_ldr, args)
    # 0.9063
    # SmothL1
