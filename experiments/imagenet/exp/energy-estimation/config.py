"""
-*- coding: utf-8 -*-

@Time    : 2021-11-03 16:06

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : vgg_config.py
"""
import os
import argparse


parser = argparse.ArgumentParser(description='Test SNN Checkpoint')
parser.add_argument('--data', metavar='DIR', default='/data2/wzm/imagenet',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--resume',

                    default='../../best_model/ann_resnet34_act_nqc-t_4_epoch2_100_lr2_0.0003_pat_10_512_128_cas_3.pth'
                    # default='../../best_model/ann_resnet34_act_nqc-t_64_epoch2_100_lr2_0.001_pat_10_128_128_cas_1.pth'
                    # default='./snn_checkpoint/ann_resnet34_act_nqc-t_8_epoch2_100_lr2_0.0003_pat_10_512_128_cas_1.pth'
                    # default='./snn_checkpoint/ann_resnet34_act_nqc-t_16_epoch2_100_lr2_0.0003_pat_10_256_128.pth'
                    # default='../../best_model/ann_vgg16_act_nqc-t_64_epoch2_100_lr2_0.0003_pat_10_128_128_cas_1.pth'
                    # default='./checkpoint/imagenet_resnet34_act_nqc_wd_0.0001_lr_0.001_t_16_cas_1-model_best.pth.tar'
                    , type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    # choices=model_names,
                    help='model architecture: ' +
                         # ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--T', default=None, type=int,
                    help='number of time steps.')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--aug', action='store_true',
                    help='use color jitter for augmentation')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--acts_bz', default=-1, type=int)  # todo: never change as it's a finetune method

parser.add_argument('--seed', default=80, type=int,
                    help='random seed for reproduce')



# parser = argparse.ArgumentParser(description='Test the functions between ReLU and Clamp')
# parser.add_argument('--seed', default=0, type=int, help="random seed")
# parser.add_argument('--pretrain',
#                     default='./checkpoint/cifar10_cut_aug-ann_vgg16_act_clamp-opt_sgdm_bn_bias_wd_0.0005_cas_1.pth',
#                     type=str, help='pretrained ann model path')
# parser.add_argument('--act_fun_name', default='noisy_qcs', type=str)
# parser.add_argument('--architecture', default="ResNet18_modified", type=str, help="VGG5|VGG9|VGG11|VGG13|VGG16|"
#                                                                          "VGG19|CIFAR|ALEX")
# parser.add_argument('--bias', default=False, action='store_true')
# parser.add_argument('--use_bn', default=False, action='store_true')
#
# parser.add_argument('--enable_shift', default=False, action='store_true')
# parser.add_argument('--lr_interval', default=[0.2, 0.8, 0.9], type=float, nargs='+',
#                     help='a relative point for lr reduction')
#
# parser.add_argument('--dataset', default="CIFAR10", type=str, help="CIFAR10|CIFAR100|MNIST")
# parser.add_argument('--data_path', default="/data1T/wzm/cifar10", type=str)
#
# parser.add_argument('--train_batch_size', default=128, type=int)
# parser.add_argument('--val_batch_size', default=128, type=int)
# parser.add_argument('--record_interval', default=2, type=int)
# parser.add_argument('--device_name', default='cuda:1', type=str)
#
# parser.add_argument('--T', default=8, type=int, help='number of timesteps')
# parser.add_argument('--num_samples', default=1024, type=int)
#
# parser.add_argument('--acts_bz', default=-1, type=int)  # todo: never change as it's a finetune method

args = parser.parse_known_args()[0]
defaults = parser.parse_args([])

args.cmd = [' --' + str(k) + ' ' + str(v) for k, v in set(vars(args).items()) - set(vars(defaults).items())]
#
