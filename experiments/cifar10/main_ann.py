"""
-*- coding: utf-8 -*-

@Time    : 2021-10-02 10:37

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : main_ann.py
"""
import sys

sys.path.append('.')
sys.path.append("..")
sys.path.append("../..")
print(sys.path)
# todo: sys.path.append 仅适用于测试, 实际调用应在项目根路径下将测试代码作为模块调用
from models.vgg import SpikingVGG, rVGG, cfg, getMaxAct, replaceAct
from models.resnet import ResNetX
from models.LeNet import LeNet5
from models.mobilenet import MobileNetV2
from models.resnetxt import ResNeXt29_2x64d

import torch
import os
import csv
import re
import time
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from util.transform import AddGaussianNoise
from util.image_augment import CIFAR10Policy, Cutout
from util.util import setup_seed, Logger, transfer_model
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from model.surrogate_act import Clamp, Quantize, TrainableQCS, TClamp, NoisyQC, NoisyQCS, LeackyQCS, LeackyClamp


def adjustLR(epoch, optimizer):
    global lr, lr_reduce
    if epoch in lr_interval:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_scale
            lr = param_group['lr']


def finetune(ann, pretrain=None, act_fun=Clamp,
             dst_act_fun=None):  # todo: 迁移到args中
    global train_ldr, val_ldr, cvt_act_fun
    if pretrain is None:
        return ann
    assert ann.hook, 'hook must be set as True when finetune with adptive QC-Train '
    state = torch.load(pretrain, map_location=device)['best_net']
    transfer_dict = transfer_model(state, ann.state_dict(), model=False)
    ann.load_state_dict(transfer_dict, strict=False)
    ann.to(device, dtype)
    max_act = getMaxAct(ann, train_ldr, device, dtype)
    # max_act = torch.ones(num_perserve)
    # max_act = nn.Parameter(torch.tensor(max_act))
    if dst_act_fun is None:
        cvt_act_fun = [nn.Sequential(Quantize(factor=T / act), Clamp(max=act)) for act in max_act]
    # cvt_act_fun = [nn.Sequential(Quantize(factor=T), Clamp(max=act)) for act in max_act]
    # cvt_act_fun = [ Clamp(max=act)) for act in max_act]
    print('Pretrained:', runTest(val_ldr, ann, nn.CrossEntropyLoss()))
    ann = replaceAct(ann, act_fun, cvt_act_fun)
    print('Converted:', runTest(val_ldr, ann, nn.CrossEntropyLoss()))
    return ann


def runTrain(epoch, train_ldr, optimizer, model, evaluator, args=None, encoder=None):
    loss_record = []
    predict_tot = []
    label_tot = []
    model.train()
    start_time = time.time()
    for idx, (ptns, labels) in enumerate(train_ldr):
        ptns, labels = ptns.to(args.device), labels.to(args.device)
        if encoder is not None:
            ptns = encoder(ptns)
        # print(labels.dtype,labels.shape)
        optimizer.zero_grad()
        output = model(ptns)
        # target = torch.nn.functional.one_hot(labels)
        # print(target.shape)
        loss = evaluator(output, labels)
        loss.backward()
        # for param in snn.named_parameters():
        #     if not param[1].grad is None:
        #         print(param[0],  torch.norm(torch.tensor(param[1].grad)))
        # for param in snn.named_parameters():
        #     if not param[1].grad is None:
        #         print(param[0],  torch.norm(torch.tensor(param[1].grad)))
        optimizer.step()
        predict = torch.argmax(output, axis=1)
        # record results
        loss_record.append(loss.detach().cpu())
        # print(output)
        # print(torch.mean((predict==labels).float()))
        # print(predict)
        predict_tot.append(predict)
        # print(predict.shape)
        label_tot.append(labels)
        # log
        if (idx + 1) % args.log_interval == 0:
            print('\nEpoch [%d/%d], Step [%d/%d], Loss: %.5f'
                  % (epoch, args.num_epoch + args.start_epoch, idx + 1, len(train_ldr.dataset) // args.train_batch_size,
                     loss_record[-1] / args.train_batch_size))
            running_loss = 0
            print('Time elasped:', time.time() - start_time)
    predict_tot = torch.cat(predict_tot)
    label_tot = torch.cat(label_tot)
    train_acc = torch.mean((predict_tot == label_tot).float())
    train_loss = torch.tensor(loss_record).sum() / len(label_tot)

    return train_acc.item(), train_loss


def runTest(val_ldr, model, evaluator, args=None, encoder=None):
    model.eval()
    with torch.no_grad():
        predict_tot = {}
        label_tot = []
        loss_record = []
        key = 'ann' if encoder is None else 'snn'
        for idx, (ptns, labels) in enumerate(val_ldr):
            # ptns: batch_size x num_channels x T x nNeu ==> batch_size x T x (nNeu*num_channels)
            ptns, labels = ptns.to(args.device), labels.to(args.device)
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
            val_acc = torch.mean((predict_tot.cpu() == label_tot.cpu()).float()).item()
        return val_acc, val_loss


def loadData(name, root, cutout=False, auto_aug=False):
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


def regular_set(model, paras=([], [], [])):
    if model is None:
        return paras
    for n, module in model._modules.items():
        if module is None: continue
        if isinstance(module, (NoisyQC, TClamp, NoisyQCS)) and hasattr(module, "thresh"):
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


def main(args):
    # todo: add learning rate scheduler [finish]
    # todo: add a data enhancement here [finish] ( auto_aug & cutoff )
    globals().update(vars(args))
    dropout = classifier_drop
    lr = init_lr
    # initial varaibles both for record and visulization
    args.start_epoch = start_epoch = 0
    args.start_acc = best_acc = 0
    train_trace, val_trace = dict(), dict()
    train_trace['acc'], train_trace['loss'] = [], []
    val_trace['acc'], val_trace['loss'] = [], []
    writer = SummaryWriter(log_path)
    train_data, val_data, num_class = loadData(dataset, data_path, cutout=cutout, auto_aug=auto_aug)
    train_ldr = torch.utils.data.DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True,
                                            pin_memory=True, num_workers=4)
    val_ldr = torch.utils.data.DataLoader(dataset=val_data, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                          num_workers=4)
    if 'VGG' in args.architecture:
        model = rVGG(architecture, num_class, dataset, act_fun=act_fun(**act_fun_args), bias=bias,
                     feature_drop=feature_drop,
                     dropout=dropout,
                     use_bn=use_bn, hook=True).to(device, dtype)
    elif 'Res' in args.architecture:
        depth = int(re.findall("\d+", args.architecture)[0])
        model = ResNetX(depth, num_class=num_class, act_fun=act_fun(**act_fun_args),
                        modified='modified' in args.architecture).to(device, dtype)
    elif 'LeNet' in args.architecture:
        assert args.dataset == 'MNIST', 'LeNet only use for MNIST'
        model = LeNet5(act_fun=act_fun(**act_fun_args)).to(device, dtype)
    elif 'mobilev2' == args.architecture:
        model = MobileNetV2(width_mult=1, n_class=10, input_size=32, act_fun=act_fun(**act_fun_args)).to(device, dtype)
    elif 'resnetxt' == args.architecture:
        model = ResNeXt29_2x64d(act_fun=act_fun(**act_fun_args)).to(device, dtype)
    evaluator = torch.nn.CrossEntropyLoss()
    if (pretrain is not None):
        # state = torch.load(os.path.join(ckpt_path, model_name + '.t7'))
        state = torch.load(os.pnath.join(pretrain))
        model.load_state_dict(state['best_net'])
        val_acc, val_loss = runTest(val_ldr, model, evaluator, args=args)
        print('Load checkpoint from {} with Acc. : {}'.format(pretrain, val_acc))
        args.start_epoch = start_epoch = state['best_epoch']
        train_trace = state['traces']['train']
        val_trace = state['traces']['val']
        best_acc = state['best_acc']
        if args.discard:
            best_acc = 0
            print('discarding')
    # model = finetune(model, pretrain=pretrain, act_fun=act_fun)
    print(model)
    params = model.parameters()
    print(params)
    if args.split:
        params = regular_set(model)
        params = [
            {'params': params[0], 'weight_decay': 0},
            {'params': params[1], 'weight_decay': args.weight_decay},
            {'params': params[2], 'weight_decay': 0}
        ]
    optimizer = torch.optim.SGD(params, init_lr, momentum=0.9,
                                weight_decay=args.weight_decay)
    if (optim_name == 'ADAM'):
        optimizer = optim.AdamW(params, lr=init_lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epoch)
    best_epoch = 0
    for epoch in tqdm(range(start_epoch, start_epoch + num_epoch)):
        train_acc, train_loss = runTrain(epoch, train_ldr, optimizer, model, evaluator, args=args)
        # adjustLR(epoch, optimizer)
        scheduler.step()
        val_acc, val_loss = runTest(val_ldr, model, evaluator, args=args)
        # saving checkpoint
        print('validation record:', val_loss, val_acc)
        if (val_acc > best_acc):
            best_acc = val_acc
            best_epoch = epoch
            print('Saving model..  with acc {0} in the epoch {1}'.format(best_acc, epoch))
            state = {
                'best_acc': best_acc,
                'best_epoch': epoch,
                'best_net': model.state_dict(),
                'traces': {'train': train_trace, 'val': val_trace},
                'cvt_fun': cvt_act_fun,
                'config': args
            }
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

    results = {'best_acc': best_acc, 'best_epoch': best_epoch, 'model_name': model_name}
    if args.grid_search is not None:
        write_head = not os.path.exists(args.grid_search)
        with open(args.grid_search, 'a+') as f:
            writer = csv.DictWriter(f, fieldnames=list(results.keys()))
            if write_head: writer.writeheader()
            # writer.writeheader()
            writer.writerow(results)


def get_model_name(model_name, args):
    globals().update(vars(args))
    aug_str = '_'.join(['cut' if cutout else ''] + ['aug' if auto_aug else ''])
    if aug_str[0] != '_': aug_str = '_' + aug_str
    if aug_str[-1] != '_': aug_str = aug_str + '-'
    model_name += dataset.lower() + aug_str + 'ann' + '_' + architecture.lower() + '_act_' + act_fun_name + '-opt_' + optim_name.lower() + (
        '_bn' if use_bn else '') + ('_bias' if bias else '') + '_wd_' + str(
        weight_decay)
    model_name += '_t_' + str(args.T)
    cas_num = len([one for one in os.listdir(log_path) if one.startswith(model_name)])
    model_name += '_cas_' + str(cas_num)
    print('model name: ' + model_name)
    return model_name


if __name__ == '__main__':
    # global config
    # todo: try to include those parameters into a specified file [finish]
    from vgg_config import *

    # set random seed, device, data type
    setup_seed(args.seed)
    dtype = torch.float
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.dtype = dtype
    # args.device = device

    # runtime params
    best_acc = 0
    start_epoch = 0
    args.start_epoch = start_epoch
    args.log_interval = 100
    # practical config for learning scheduler
    momentum = 0.9
    lr_scale = 0.1
    args.lr_interval = [int(ratio * args.num_epoch) for ratio in args.lr_interval]

    # perserved params for the replaced activation function
    cvt_act_fun = None
    act_fun_args = {}
    if args.act_fun_name == 'relu':
        act_fun = nn.ReLU
    elif args.act_fun_name == 'clamp':
        act_fun = Clamp
    elif args.act_fun_name == 'qcs':
        act_fun = TrainableQCS
        act_fun_args['thresh'] = torch.tensor(1.0)
        act_fun_args['nb_step'] = args.T
    elif args.act_fun_name == 'tclamp':
        act_fun = TClamp
    elif args.act_fun_name == 'nqc':
        act_fun = NoisyQC
        act_fun_args = {
            'thresh': torch.tensor(10.0),
            'nb_steps': args.T,
            'noise_prob': 0.5
        }
    elif args.act_fun_name == 'nqcs':
        act_fun = NoisyQCS
        act_fun_args = {'thresh': torch.tensor(1.0), 'nb_step': args.T, 'shift': 0, 'trainable': True,
                        'p': args.denoise_rate, 'detach': args.detach}
    elif args.act_fun_name == 'lqcs':
        act_fun = LeackyQCS
        act_fun_args = {
            'thresh': torch.tensor(5.0),
            'nb_steps': args.T,
        }
    elif args.act_fun_name == 'lclamp':
        act_fun = LeackyClamp
    elif args.act_fun_name == 'lrelu':
        act_fun = nn.LeakyReLU

    model_name = get_model_name('', args)
    args.log_path = os.path.join(args.log_path, model_name)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    sys.stdout = Logger(os.path.join(args.log_path, model_name + '.log'))
    sys.stderr = Logger(os.path.join(args.log_path, model_name + '.log'))
    # test git
    # print(model_name)
    # act_fun_name = 'c'
    # T = 16
    # if (act_fun_name == 'c'):
    #     model_name = 'c_'
    #     act_fun = Clamp
    # act_fun_name = 'cq'
    # if (act_fun_name == 'cq'):
    #     T = 256
    #     model_name = 'cq_'
    #     act_fun = nn.Sequential(Clamp(), Quantize(T, method=2))

    # redirect the output
    main(args)
