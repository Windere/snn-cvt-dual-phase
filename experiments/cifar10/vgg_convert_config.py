"""
-*- coding: utf-8 -*-

@Time    : 2021-11-03 16:06

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : vgg_config.py
"""
import os
import argparse

parser = argparse.ArgumentParser(description='Convert a Pretrained ANNs into SNNs with Dual Phases')
parser.add_argument('--seed', default=0, type=int, help="random seed")
parser.add_argument('--pretrain_path',
                    default='./checkpoint/cifar10_cut_aug-ann_res18_act_tclamp-opt_sgdm_bn_bias_wd_0.0005_cas_3.pth',
                    type=str, help='pretrained ann model path')
parser.add_argument('--act_fun_name', default='tclamp', type=str)
parser.add_argument('--architecture', default="VGG16", type=str, help="VGG5|VGG9|VGG11|VGG13|VGG16|"
                                                                      "VGG19|CIFAR|ALEX")
parser.add_argument('--bias', default=False, action='store_true')
parser.add_argument('--feature_drop', default=0, type=float)
parser.add_argument('--classifier_drop', default=0, type=float)
parser.add_argument('--use_bn', default=False, action='store_true')
parser.add_argument('--enable_shift', default=False, action='store_true')
parser.add_argument('--lr_interval', default=[0.2, 0.8, 0.9], type=float, nargs='+',
                    help='a relative point for lr reduction')

parser.add_argument('--dataset', default="CIFAR10", type=str, help="CIFAR10|CIFAR100|MNIST")
parser.add_argument('--data_path', default="/data1T/wzm/cifar10", type=str)
parser.add_argument('--ckpt_path', default="./cvt_checkpoint", type=str, help="checkpoint path")
parser.add_argument('--log_path', default="./cvt_log", type=str, help="log path")
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--val_batch_size', default=128, type=int)
parser.add_argument('--record_interval', default=2, type=int)
parser.add_argument('--warmup_step', default=0, type=int)
parser.add_argument('--device_name', default='cuda:0', type=str)

parser.add_argument('--denoise_rate', default=0.1, type=float, help='the probability ')
parser.add_argument('--T', default=8, type=int, help='number of timesteps')
parser.add_argument('--num_epoch1', default=100, type=int)
parser.add_argument('--lr1', default=1e-4, type=float)
parser.add_argument('--wd1', default=0, type=float)
parser.add_argument('--detach', default=False, action='store_true')

parser.add_argument('--optim_name', default='SGDM', type=str)  # todo: never change as it's a finetune method
parser.add_argument('--num_epoch2', default=10000, type=int)
parser.add_argument('--patience', default=100, type=int)  # todo: never change as it's a finetune method
parser.add_argument('--num_samples', default=1024, type=int)

parser.add_argument('--wd2', default=0, type=float)
parser.add_argument('--bz2', default=128, type=int)
parser.add_argument('--lr2', default=1e-4, type=float)
parser.add_argument('--grid_search', default='./summary.csv', type=str)

# parser.add_argument('--grid_search', default=True, action='store_true')
parser.add_argument('--acts_bz', default=-1, type=int)  # todo: never change as it's a finetune method
parser.add_argument('--optim_name2', default='ADAM', type=str)  # todo: never change as it's a finetune method

# args.num_samples = 1024
# args.wd2 = 0
# args.lr2 = 0.01
# args.num_epoch2 = 1000
# best_acc_rec = 0
# lr2_rec = 0

# parser.add_argument('--bias', default=True, type=bool)
# parser.add_argument('--num_epoch', default=300, type=int)
# parser.add_argument('--act_fun_name', default='relu', type=str)
# parser.add_argument('--lr_interval', default=[0.2, 0.8, 0.9], type=float, nargs='+',
#                     help='a relative point for lr reduction')
# parser.add_argument('--scheduler', default='COSINE')
# args = parser.parse_args()
args = parser.parse_known_args()[0]
if not args.data_path.startswith('.') and not args.data_path.startswith('/'):
    path_list = args.data_path.split('/')
    root = os.environ.get(path_list[0])
    args.data_path = os.path.join(root, os.path.join(*path_list[1:]))
# dataset = args.dataset
# architecture = args.model
# data_path = args.data_path
# ckpt_path = args.ckpt_path
# log_path = args.log_path
# cutout = args.cutout
# auto_aug = args.autoaugment
# pretrain = args.pretrain
# train_batch_size = args.train_batch_size
# val_batch_size = args.val_batch_size
# lr = args.init_lr
# feature_drop = args.feature_drop
# dropout = args.classifier_drop
# use_bn = args.use_bn
# weight_decay = args.weight_decay
# bias = args.bias
# num_epoch = args.num_epoch
# optim_name = args.optimizer
# lr_interval = args.lr_interval
# scheduler = args.scheduler
defaults = parser.parse_args([])
args.lr_interval = tuple(args.lr_interval)
defaults.lr_interval = tuple(defaults.lr_interval)

args.cmd = [' --' + str(k) + ' ' + str(v) for k, v in set(vars(args).items()) - set(vars(defaults).items())]
if not os.path.exists(args.ckpt_path):
    os.mkdir(args.ckpt_path)
if not os.path.exists(args.log_path):
    os.mkdir(args.log_path)
