"""
-*- coding: utf-8 -*-

@Time    : 2021-11-03 16:06

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : vgg_config.py
"""
import os
import argparse

parser = argparse.ArgumentParser(description='Training Source VGG Net with ReLU during Conversion')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--architecture', default="VGG16", type=str, help="VGG5|VGG9|VGG11|VGG13|VGG16|"
                                                                      "VGG19|CIFAR|ALEX")
parser.add_argument('--dataset', default="CIFAR10", type=str, help="CIFAR10|CIFAR100|MNIST")
parser.add_argument('--data_path', default="data1/cifar10", type=str)
parser.add_argument('--ckpt_path', default="./checkpoint", type=str, help="checkpoint path")
parser.add_argument('--log_path', default="./log", type=str, help="log path")
parser.add_argument('--auto_aug', default=False, action='store_true')
parser.add_argument('--cutout', default=False, action='store_true')
parser.add_argument('--pretrain', default=None)
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--val_batch_size', default=128, type=int)
parser.add_argument('--init_lr', default=0.1, type=float)
parser.add_argument('--feature_drop', default=0, type=float)
parser.add_argument('--classifier_drop', default=0 , type=float)
# parser.add_argument('--use_bn', default=True, type=bool)
# parser.add_argument('--bias', default=True, type=bool)
parser.add_argument('--use_bn', default=False, action='store_true')
parser.add_argument('--bias', default=False, action='store_true')

parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--num_epoch', default=300, type=int)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--optim_name', default='SGDM', type=str)
parser.add_argument('--act_fun_name', default='relu', type=str)
parser.add_argument('--T', default=8, type=int, help='only used in qcs activation')
parser.add_argument('--lr_interval', default=[0.2, 0.8, 0.9], type=float, nargs='+',
                    help='a relative point for lr reduction')
parser.add_argument('--scheduler', default='COSINE')

parser.add_argument('--denoise_rate', default=0.1, type=float)
parser.add_argument('--detach', default=False, action='store_true')
parser.add_argument('--split', default=False, action='store_true')
parser.add_argument('--grid_search', default='./summary.csv', type=str)
parser.add_argument('--discard', default=False, action='store_true')

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
if not os.path.exists(args.ckpt_path):
    os.mkdir(args.ckpt_path)
if not os.path.exists(args.log_path):
    os.mkdir(args.log_path)
print(args)
