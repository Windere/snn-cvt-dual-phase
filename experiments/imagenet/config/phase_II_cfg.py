"""
-*- coding: utf-8 -*-

@Time    : 2022/1/8 21:03

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : phase_II_cfg.py
"""
import os
import argparse

parser = argparse.ArgumentParser(description='Second Phase in ImageNet')
parser.add_argument('--act_fun_name', default='relu', type=str)

parser.add_argument('--seed', default=0, type=int, help="random seed")
parser.add_argument('--T', default=32, type=int, help='number of timesteps')
parser.add_argument('--device_name', default="cuda", type=str, help="device")
parser.add_argument('--pretrain',
                    default='./checkpoint/fintune_imagenet_resnet34_act_noisy_qcs-opt_adam_wd_0.0_cas_2-checkpoint-module.pth.tar',
                    type=str, help='pretrained qc ann model path')
parser.add_argument('-p', '--print_interval', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--enable_shift', default=False, action='store_true')
parser.add_argument('--data_path', default="/data2/wzm/imagenet", type=str)
parser.add_argument('--ckpt_path', default="./snn_checkpoint", type=str, help="checkpoint path")
parser.add_argument('--log_path', default="./snn_log", type=str, help="log path")

parser.add_argument('--train_batch_size', default=1024, type=int)
parser.add_argument('--val_batch_size', default=500, type=int)
parser.add_argument('--denoise_rate', default=0.1, type=float)
parser.add_argument('--lr1', default=1e-4, type=float)
parser.add_argument('--wd1', default=0, type=float)
parser.add_argument('--optim_name1', default='SGDM', type=str)  # todo: never change as it's a finetune method
parser.add_argument('--num_epoch1', default=100, type=int)

parser.add_argument('--wd2', default=0, type=float)
parser.add_argument('--bz2', default=32, type=int)
parser.add_argument('--lr2', default=3e-4, type=float)
parser.add_argument('--num_epoch2', default=1000, type=int)
parser.add_argument('--patience', default=100, type=int)  # todo: never change as it's a finetune method
parser.add_argument('--num_samples', default=256, type=int)
parser.add_argument('--acts_bz', default=-1, type=int)  # todo: never change as it's a finetune method
parser.add_argument('--warmup_step', default=0, type=int)  # todo: never change as it's a finetune method
parser.add_argument('--optim_name2', default='ADAM', type=str)  # todo: never change as it's a finetune method

args = parser.parse_known_args()[0]
if not os.path.exists(args.ckpt_path):
    os.mkdir(args.ckpt_path)
if not os.path.exists(args.log_path):
    os.mkdir(args.log_path)
args = parser.parse_known_args()[0]

defaults = parser.parse_args([])

args.cmd = [' --' + str(k) + ' ' + str(v) for k, v in set(vars(args).items()) - set(vars(defaults).items())]
