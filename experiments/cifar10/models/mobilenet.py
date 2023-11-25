"""
-*- coding: utf-8 -*-

@Time    : 2023/6/21 17:09

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : mobilenet.py
"""
import torch.nn as nn
import math
import copy
from model.snn import LIFCell
from .snn import IsomorphicSNN


def conv_bn(inp, oup, stride, act_fun=nn.ReLU6(inplace=True)):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        copy.deepcopy(act_fun)
    )


def conv_1x1_bn(inp, oup, act_fun=nn.ReLU6(inplace=True)):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        copy.deepcopy(act_fun)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, act_fun=nn.ReLU6(inplace=True)):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # nn.ReLU6(inplace=True),
                copy.deepcopy(act_fun),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                copy.deepcopy(act_fun),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                copy.deepcopy(act_fun),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        # add a ReLU to replace with IF neuron
        self.fw = copy.deepcopy(act_fun)

    def forward(self, x):
        if self.use_res_connect:
            return self.fw(x + self.conv(x))
        else:
            return self.fw(self.conv(x))


class SpikeInvertedResidual(nn.Module):
    num_layer = 2

    # todo: try to support the BN conversion
    def __init__(self, block, thresh, decay, init_mem, shift, **kwargs_lif):
        super(SpikeInvertedResidual, self).__init__()
        self.spike_conv = []
        self.use_res_connect = block.use_res_connect
        i = 0
        # print(block)
        for module in block.conv.modules():
            if isinstance(module, nn.Conv2d):
                self.spike_conv.append(LIFCell(module, thresh=thresh[i], decay=decay[i], shift=shift[i],
                                               init_mem=init_mem[i], **kwargs_lif))
                i += 1
        self.stright_conv = nn.Sequential(*self.spike_conv[:-1])
        self.merge_conv = self.spike_conv[-1]

        # self.conv1 = LIFCell(block.conv1, thresh=thresh[0], decay=decay[0], shift=shift[0],
        #                      init_mem=init_mem[0], **kwargs_lif)
        # self.conv2 = LIFCell(block.conv2, thresh=thresh[1], decay=decay[1], shift=shift[1],
        #                      init_mem=init_mem[1], **kwargs_lif)
        # self.conv2.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        if self.use_res_connect:
            out = self.stright_conv(x)
            return self.merge_conv(out, x)
        else:
            return self.merge_conv(self.stright_conv(x))


def cvtFn(iso_snn: IsomorphicSNN, layer_idx, raw_module=None):
    kwargs = {'mode': iso_snn.mode, 'spike_fn': iso_snn.spike_fn}
    num_layer = len([m for m in raw_module.modules() if isinstance(m, nn.Conv2d)])  # todo:
    SpikeInvertedResidual.num_layer = num_layer
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
    # out = self.conv2(out, self.conv2.scale * self.shortcut(x))


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., act_fun=nn.ReLU6(inplace=True)):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        self.act_fun = act_fun
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 1, act_fun=act_fun)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, act_fun=act_fun))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, act_fun=act_fun))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel, act_fun=act_fun))
        self.features.append(nn.AdaptiveAvgPool2d(1))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        # x = x.mean(3).mean(2)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(pretrained=True):
    model = MobileNetV2(width_mult=1)

    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        model.load_state_dict(state_dict)
    return model


mobilenet_specials = {InvertedResidual: (SpikeInvertedResidual, cvtFn)}

if __name__ == '__main__':
    net = MobileNetV2(width_mult=1, n_class=10, input_size=32, act_fun=nn.ReLU6())
    print(net)
