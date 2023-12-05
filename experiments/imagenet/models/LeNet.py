"""
-*- coding: utf-8 -*-

@Time    : 2021/12/13 17:28

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : LeNet.py
"""
import copy
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, act_fun=nn.ReLU()):
        super().__init__()
        self.act_fun = copy.deepcopy(act_fun)
        self.feature = nn.Sequential(nn.Conv2d(1, 6, 5, padding=2), nn.BatchNorm2d(6), copy.deepcopy(act_fun),
                                     nn.AvgPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(6, 16, 5), nn.BatchNorm2d(16), copy.deepcopy(act_fun),
                                     nn.AvgPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(nn.Linear(16 * 5 * 5, 120), copy.deepcopy(act_fun),
                                        nn.Linear(120, 84), copy.deepcopy(act_fun),
                                        nn.Linear(84, 10))

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.classifier(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
