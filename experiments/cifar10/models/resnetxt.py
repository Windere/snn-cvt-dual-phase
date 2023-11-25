"""
-*- coding: utf-8 -*-

@Time    : 2023/6/22 1:09

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : resnetxt.py
"""
'''ResNeXt in PyTorch.

See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.snn import LIFCell
from .snn import IsomorphicSNN


class SpikeXtBlock(nn.Module):
    num_layer = 3

    # todo: try to support the BN conversion
    def __init__(self, block, thresh, decay, init_mem, shift, **kwargs_lif):
        super(SpikeXtBlock, self).__init__()
        self.conv1 = LIFCell(block.conv1, thresh=thresh[0], decay=decay[0], shift=shift[0],
                             init_mem=init_mem[0], **kwargs_lif)
        self.conv2 = LIFCell(block.conv2, thresh=thresh[1], decay=decay[1], shift=shift[1],
                             init_mem=init_mem[1], **kwargs_lif)
        self.conv3 = LIFCell(block.conv3, thresh=thresh[2], decay=decay[2], shift=shift[2],
                             init_mem=init_mem[2], **kwargs_lif)
        # self.conv2.scale = nn.Parameter(torch.tensor(1.0))
        self.shortcut = block.shortcut

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # out = self.conv2(out, self.conv2.scale * self.shortcut(x))
        out = self.conv3(out, self.shortcut(x))
        return out


def cvtFn(iso_snn: IsomorphicSNN, layer_idx, raw_module=None):
    kwargs = {'mode': iso_snn.mode, 'spike_fn': iso_snn.spike_fn}
    num_layer = SpikeXtBlock.num_layer  # todo:
    kwargs['thresh'] = iso_snn.thresh[layer_idx:layer_idx + num_layer]
    kwargs['decay'] = iso_snn.decay[layer_idx:layer_idx + num_layer]
    init_mem = []
    shift = []
    for i in range(len(kwargs['thresh'])):
        if iso_snn.enable_shift:
            shift.append(abs(kwargs['thresh'][i]) / (2 * iso_snn.nb_steps))  # todo: 注意修正
        else:
            shift.append(0)
        init_mem.append(abs(kwargs['thresh'][i]) / 2 if iso_snn.enable_init_volt else 0)
        # pass
    kwargs['init_mem'] = init_mem
    kwargs['shift'] = shift
    # layer_idx += num_layer
    return kwargs


class XtBlock(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1, act_fun=nn.ReLU()):
        super(XtBlock, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.act_fun_1 = copy.deepcopy(act_fun)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.act_fun_2 = copy.deepcopy(act_fun)
        self.conv3 = nn.Conv2d(group_width, self.expansion * group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * group_width)
            )
        self.act_fun_3 = copy.deepcopy(act_fun)

    def forward(self, x):
        out = self.act_fun_1(self.bn1(self.conv1(x)))
        out = self.act_fun_2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act_fun_3(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10, act_fun=nn.ReLU()):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.act_fun = act_fun
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act_fun_1 = copy.deepcopy(self.act_fun)  # act_fun is a perserved word for replaceAct.
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        self.linear = nn.Linear(cardinality * bottleneck_width * 8, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                XtBlock(self.in_planes, self.cardinality, self.bottleneck_width, stride, act_fun=self.act_fun))
            self.in_planes = XtBlock.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act_fun_1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNeXt29_2x64d(act_fun=nn.ReLU()):
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=2, bottleneck_width=64, act_fun=act_fun)


def ResNeXt29_4x64d(act_fun=nn.ReLU()):
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=4, bottleneck_width=64, act_fun=act_fun)


def ResNeXt29_8x64d(act_fun=nn.ReLU()):
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=8, bottleneck_width=64, act_fun=act_fun)


def ResNeXt29_32x4d(act_fun=nn.ReLU()):
    return ResNeXt(num_blocks=[3, 3, 3], cardinality=32, bottleneck_width=4, act_fun=act_fun)


def test_resnext():
    net = ResNeXt29_2x64d(nn.ReLU6())
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())


resnetxt_specials = {XtBlock: (SpikeXtBlock, cvtFn)}

if __name__ == '__main__':
    test_resnext()
