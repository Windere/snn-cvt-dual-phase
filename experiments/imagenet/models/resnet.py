'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
    https://github.com/junyuseu/pytorch-cifar-models/blob/master/models/resnet_cifar.py
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.snn_layer import LIFCell
from model.snn import LIFCell
from .snn import IsomorphicSNN
from util.fold_bn import ScaleStraightThrough
import math


class BasicBlock(nn.Module):
    expansion = 1
    num_layer = 2

    def __init__(self, in_planes, planes, stride=1, act_fun=nn.ReLU(), use_bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act_fun_1 = copy.deepcopy(act_fun)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(planes)
        # todo: combine both sequential and scale
        self.shortcut = ScaleStraightThrough()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=use_bias),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.act_fun_2 = copy.deepcopy(act_fun)

    def forward(self, x):
        out = self.act_fun_1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act_fun_2(out)
        return out


class SpikeBasicBlock(nn.Module):
    num_layer = 2

    # todo: try to support the BN conversion
    def __init__(self, block, thresh, decay, init_mem, shift, **kwargs_lif):
        super(SpikeBasicBlock, self).__init__()
        self.conv1 = LIFCell(block.conv1, thresh=thresh[0], decay=decay[0], shift=shift[0],
                             init_mem=init_mem[0], **kwargs_lif)
        self.conv2 = LIFCell(block.conv2, thresh=thresh[1], decay=decay[1], shift=shift[1],
                             init_mem=init_mem[1], **kwargs_lif)
        # self.conv2.scale = nn.Parameter(torch.tensor(1.0))
        self.shortcut = block.shortcut

    def forward(self, x):
        out = self.conv1(x)
        # out = self.conv2(out, self.conv2.scale * self.shortcut(x))
        out = self.conv2(out, self.shortcut(x))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_bias=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=use_bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=use_bias)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=use_bias),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def cvtFn(iso_snn: IsomorphicSNN, layer_idx):
    kwargs = {'mode': iso_snn.mode, 'spike_fn': iso_snn.spike_fn}
    num_layer = SpikeBasicBlock.num_layer  # todo:
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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_class=10, act_fun=nn.ReLU(), use_bias=False):
        super(ResNet, self).__init__()
        # todo: update as a much elegant method for replaceAct
        self.use_bias = use_bias
        self.act_fun = act_fun
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=use_bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.act_fun_1 = copy.deepcopy(self.act_fun)  # act_fun is a perserved word for replaceAct.
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_class)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, act_fun=self.act_fun, use_bias=self.use_bias))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(self.bn1(self.conv1(x)).shape)
        # print(self.act_fun_1)
        out = self.pool(self.act_fun_1(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_Modified(nn.Module):
    def __init__(self, block, num_blocks, num_class=10, act_fun=nn.ReLU()):
        super(ResNet_Modified, self).__init__()
        # todo: update as a much elegant method for replaceAct
        self.act_fun = act_fun
        self.in_planes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            copy.deepcopy(self.act_fun),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            copy.deepcopy(self.act_fun),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.act_fun_1 = copy.deepcopy(self.act_fun)  # act_fun is a perserved word for replaceAct.
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_class)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, act_fun=self.act_fun))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act_fun_1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_Old(nn.Module):
    def __init__(self, block, num_blocks, num_class=10, act_fun=nn.ReLU()):
        super(ResNet_Old, self).__init__()
        # todo: update as a much elegant method for replaceAct
        self.act_fun = act_fun
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act_fun_1 = copy.deepcopy(self.act_fun)  # act_fun is a perserved word for replaceAct.
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_class)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, act_fun=self.act_fun))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act_fun_1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, act_fun=nn.ReLU()):
        super(ResNet_Cifar, self).__init__()
        self.act_fun = act_fun
        self.in_planes = in_planes_ = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        # self.relu = nn.ReLU(inplace=True)
        self.act_fun_1 = copy.deepcopy(self.act_fun)  # act_fun is a perserved word for replaceAct.
        self.layer1 = self._make_layer(block, in_planes_, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes_ * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes_ * 4, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes_ * 4 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, act_fun=self.act_fun))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_fun_1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


imgnet_depth_lst = [18, 34, 50, 101, 152]


def cfg(depth):
    assert (depth in imgnet_depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2, 2, 2, 2]),
        '34': (BasicBlock, [3, 4, 6, 3]),
        '50': (Bottleneck, [3, 4, 6, 3]),
        '101': (Bottleneck, [3, 4, 23, 3]),
        '152': (Bottleneck, [3, 8, 36, 3]),
    }

    return cf_dict[str(depth)]


def cifar_cfg(depth):
    depth_lst = [20, 32, 44, 56, 110, 164]
    assert (depth in depth_lst), "Error : Resnet Cifar depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '20': (BasicBlock, [3, 3, 3]),
        '32': (BasicBlock, [5, 5, 5]),
        '44': (BasicBlock, [7, 7, 7]),
        '56': (BasicBlock, [9, 9, 9]),
        '110': (BasicBlock, [18, 18, 18]),
        '164': (Bottleneck, [18, 18, 18]),
    }

    return cf_dict[str(depth)]


def ResNetX(depth, act_fun, num_class=10, modified=False, old=False, use_bias=False):
    if depth in imgnet_depth_lst:
        block, num_blocks = cfg(depth)
        if modified:
            return ResNet_Modified(block, num_blocks, num_class, act_fun=act_fun)
        else:
            if old:
                return ResNet_Old(block, num_blocks, num_class, act_fun=act_fun)
            return ResNet(block, num_blocks, num_class, act_fun=act_fun, use_bias=use_bias)

    else:
        block, num_blocks = cifar_cfg(depth)
        return ResNet_Cifar(block, num_blocks, num_class, act_fun=act_fun)


resnet_specials = {BasicBlock: (SpikeBasicBlock, cvtFn)}
