"""
-*- coding: utf-8 -*-

@Time    : 2021-10-02 9:22

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : vgg.py
"""
import torch.nn as nn
import math
import torch
import copy
from util.hooks import Hook
from model.surrogate_act import SurrogateHeaviside
from model.snn import SpikingAvgPool, LIFLayer, SpikingDropout, LIFReadout

cfg = {
    'VGG5': [64, 'A', 128, 128, 'A'],
    'VGG9': [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 'A', 512, 512, 'A', 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512, 'A'],
    'VGG16C': [64, 64, 'A', 128, 128, 'A', 256, 256, '256', 'A', 512, 512, '512', 'A', 512, 512, '512'],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512],
    'CIFAR': [128, 256, 'A', 512, 'A', 1024, 512],
    'ALEX': [96, 256, 'A', 384, 'A', 384, 256],
}


def getMaxAct(net, ldr, device, dtype):
    """
    Get the maximum activations layer by layer on dataloader ldr
    """
    assert net.hook, 'hook should be set into True to find the layer-wise maximum activation '
    max_act = None
    net.to(device, dtype)
    with torch.no_grad():
        for idx, (ptns, labels) in enumerate(ldr):
            net.eval()
            ptns = ptns.to(device, dtype)
            net(ptns)
            if max_act is None:
                max_act = [0 for i in range(len(net.record))]
            for i, acts in enumerate(net.record):
                max_act[i] = max(max_act[i], acts.max().detach().cpu())
    return max_act


# def replaceAct(net, src_act, dst_acts):
#     with torch.no_grad():
#         pre_act = None
#         i = 0
#         for l, m in enumerate(net.features.children()):
#             if isinstance(m, Hook):
#                 if isinstance(net.features[l - 1], src_act):
#                     net.features[l - 1] = dst_acts[i]
#                 i += 1
#
#         for l, m in enumerate(net.classifier.children()):
#             if isinstance(m, Hook):
#                 if isinstance(net.classifier[l - 1], src_act):
#                     net.classifier[l - 1] = dst_acts[i]
#                 i += 1
#     return net

def replaceAct(net, src_act, dst_acts):
    with torch.no_grad():
        i = 0
        for l, m in enumerate(net.features.children()):
            if isinstance(m, src_act):
                net.features[l] = dst_acts[i]
                i += 1
        for l, m in enumerate(net.classifier.children()):
            if isinstance(m, src_act):
                net.classifier[l] = dst_acts[i]
                i += 1
        return net


class rVGG(nn.Module):
    "The implement of restrict VGG net with clip , quantization, forbidden bias"

    # todo: add batch normalization layer for vgg  [finish]
    # todo: write a function fusing batch transform layer into weight and bias.

    def __init__(self, vgg_name='VGG16', num_class=10, dataset='CIFAR10', kernel_size=3, feature_drop=0.2, dropout=0.5,
                 use_bn=False,
                 act_fun=nn.ReLU(inplace=True), bias=False, hook=False, z_hook=False):
        super(rVGG, self).__init__()
        self.dataset = dataset
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.feature_drop = feature_drop
        self.act_fun = act_fun
        self.bias = bias
        self.num_class = num_class
        self.record = []
        self.z_record = []
        self.use_bn = use_bn
        self.hook = hook
        self.z_hook = z_hook
        # build up 5 blocks of VGG series
        self.features = self._make_blocks(cfg[vgg_name])
        # build up the fully connected at the end
        self.classifier = self._make_fcs(vgg_name, dataset)
        self._initialize_weights()

    def forward(self, x):
        self.record.clear()
        self.z_record.clear()
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        # print('expanding shape:', out.shape)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                # n = m.weight.size(1)
                # m.weight.data.normal_(0, 1.0 / float(n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_blocks(self, cfg):
        layers = []
        if self.dataset == 'MNIST':
            in_channels = 1
        else:
            in_channels = 3

        for x in cfg:
            stride = 1
            if x == 'A':
                layers.pop()
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                # if (self.hook):
                #     layers += [Hook(self.record)]  # add hook
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                                     stride=stride, bias=self.bias),
                           # nn.BatchNorm2d(x),
                           # nn.ReLU(inplace=True)
                           ]
                if self.use_bn:
                    layers += [nn.BatchNorm2d(x)]
                if (self.z_hook):
                    layers += [Hook(self.z_record)]
                layers += [copy.deepcopy(self.act_fun)]
                if (self.hook):
                    layers += [Hook(self.record)]  # add hook
                layers += [nn.Dropout(self.feature_drop)]
                in_channels = x

        return nn.Sequential(*layers)

    def _make_fcs(self, vgg_name, dataset):
        models = []
        if vgg_name == 'VGG5' and dataset == 'MNIST':
            models = [
                nn.Linear(128 * 7 * 7, 4096, bias=self.bias),
                copy.deepcopy(self.act_fun),
                nn.Dropout(self.dropout),
                nn.Linear(4096, 4096, bias=self.bias),
                copy.deepcopy(self.act_fun),
                nn.Dropout(self.dropout),
                nn.Linear(4096, self.num_class, bias=self.bias),
            ]
        elif vgg_name == 'VGG5' and dataset != 'MNIST':
            models = [nn.Linear(512 * 4 * 4, 4096, bias=self.bias),
                      copy.deepcopy(self.act_fun),
                      nn.Dropout(self.dropout),
                      nn.Linear(4096, 4096, bias=self.bias),
                      copy.deepcopy(self.act_fun),
                      nn.Dropout(self.dropout),
                      nn.Linear(4096, self.num_class, bias=self.bias),
                      ]
        elif (vgg_name == 'CIFAR'):
            pass
        elif (vgg_name == 'ALEX'):
            pass
        elif vgg_name != 'VGG5' and dataset == 'MNIST':
            models = [
                nn.Linear(512 * 1 * 1, 4096, bias=self.bias),
                copy.deepcopy(self.act_fun),
                nn.Dropout(self.dropout),
                nn.Linear(4096, 4096, bias=self.bias),
                copy.deepcopy(self.act_fun),
                nn.Dropout(self.dropout),
                nn.Linear(4096, self.num_class, bias=self.bias),
            ]
        elif vgg_name != 'VGG5' and dataset != 'MNIST':
            models = [nn.Linear(512, 4096, bias=self.bias),
                      copy.deepcopy(self.act_fun),
                      nn.Dropout(self.dropout),
                      nn.Linear(4096, 4096, bias=self.bias),
                      copy.deepcopy(self.act_fun),
                      nn.Dropout(self.dropout),
                      nn.Linear(4096, self.num_class, bias=self.bias)]
        hook_models = []
        for i, m in enumerate(models):
            if type(m) == type(self.act_fun):
                if self.z_hook:
                    hook_models.append(Hook(self.z_record))
                hook_models.append(m)
                if self.hook:
                    hook_models.append(Hook(self.record))
            else:
                hook_models.append(m)
        if self.hook:
            hook_models.append(Hook(self.record))
        return nn.Sequential(*hook_models)


class SpikingVGG(nn.Module):
    "The implement of Spiking VGG"

    # todo: decay, thresh initialization [finish]
    # todo: merge the online weight sharing into online dual project
    # todo: split Hook into different subclasses for different layer
    # todo: warp fcs construction into a  struction list
    # todo:  implement vgg net on other dataset
    # todo: implement a readout spiking layer in average cumulative membrane voltage
    def __init__(self, vgg_name='VGG16', num_class=10, dataset='CIFAR10', kernel_size=3, dropout=0.5, feature_drop=0.2,
                 spike_fn=SurrogateHeaviside.apply, bias=False, decay=None, thresh=None, hook=False, readout='mean',
                 neu_mode='spike'):
        super(SpikingVGG, self).__init__()
        self.dataset = dataset
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.feature_drop = feature_drop
        self.spike_fn = spike_fn
        self.vgg_name = vgg_name
        self.bias = bias
        self.num_class = num_class
        self.thresh = thresh
        self.decay = decay
        self.record = []
        self.hook = hook
        self.readout = readout
        self.neu_mode = neu_mode
        self._initialize_params()
        # build up 5 blocks of VGG series
        self.features = self._make_blocks()
        # build up the fully connected at the end
        self.classifier = self._make_fcs()
        self._initialize_weights()

    def _make_blocks(self, ann=None):
        # todo: threshhold should be consistent with CFG
        # todo: or other changes
        layers = []
        structure = cfg[self.vgg_name]
        if ann is None:
            if self.dataset == 'MNIST':
                in_channels = 1
            else:
                in_channels = 3
            lif_idx = 0
            for i, x in enumerate(structure):
                stride = 1
                # todo: add batch normalization layer for vgg and the corresponding transfer operator on snn
                if x == 'A':
                    layers.pop()
                    layers += [SpikingAvgPool(kernel_size=2, stride=2)]
                    # if (self.hook):
                    #     layers += [Hook(self.record)]  # add hook
                else:
                    func = nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                                     stride=stride, bias=self.bias)
                    layers += [
                        LIFLayer(func, thresh=self.thresh[lif_idx], decay=self.decay[lif_idx], spike_fn=self.spike_fn,
                                 mode=self.neu_mode)
                        # nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2,
                        #                  stride=stride, bias=self.bias),
                        #        # nn.ReLU(inplace=True)
                        #        copy.deepcopy(self.act_fun)
                    ]
                    lif_idx += 1
                    if (self.hook):
                        layers += [Hook(self.record)]  # add hook
                    layers += [SpikingDropout(self.feature_drop)]
                    in_channels = x
        else:
            i = 0
            for m in ann.features:
                if isinstance(m, nn.Conv2d):
                    layers += [
                        LIFLayer(m, thresh=self.thresh[i], decay=self.decay[i], spike_fn=self.spike_fn,
                                 mode=self.neu_mode)
                    ]
                    if (self.hook):
                        layers += [Hook(self.record)]  # add hook
                    i += 1
                elif isinstance(m, nn.AvgPool2d):
                    layers += [SpikingAvgPool(kernel_size=m.kernel_size, stride=m.stride)]
                    # if (self.hook):
                    #     layers += [Hook(self.record)]  # add hook
                    # i += 1
                elif isinstance(m, nn.Dropout):
                    layers += [SpikingDropout(m.p)]
        return nn.Sequential(*layers)

    def _make_fcs(self, ann=None):
        vgg_name = self.vgg_name
        dataset = self.dataset
        ilayer = len([c for c in cfg[vgg_name] if isinstance(c, int)])
        model = []
        if ann is None:
            if vgg_name == 'VGG5' and dataset == 'MNIST':
                model = [
                    LIFLayer(nn.Linear(128 * 7 * 7, 4096, bias=self.bias), thresh=self.thresh[ilayer],
                             decay=self.decay[ilayer],
                             spike_fn=self.spike_fn, mode=self.neu_mode),
                    SpikingDropout(self.dropout),
                    LIFLayer(nn.Linear(4096, 4096, bias=self.bias), thresh=self.thresh[ilayer + 1],
                             decay=self.decay[ilayer + 1],
                             spike_fn=self.spike_fn, mode=self.neu_mode),
                    SpikingDropout(self.dropout),
                    LIFReadout(nn.Linear(4096, self.num_class, bias=self.bias), thresh=self.thresh[ilayer + 2],
                               decay=self.decay[ilayer + 2],
                               spike_fn=self.spike_fn, reduction=self.readout, mode=self.neu_mode)  # todo: readout
                ]
            elif vgg_name == 'VGG5' and dataset != 'MNIST':
                model = [
                    LIFLayer(nn.Linear(512 * 4 * 4, 4096, bias=self.bias), thresh=self.thresh[ilayer],
                             decay=self.decay[ilayer],
                             spike_fn=self.spike_fn, mode=self.neu_mode),
                    SpikingDropout(self.dropout),
                    LIFLayer(nn.Linear(4096, 4096, bias=self.bias),
                             thresh=self.thresh[ilayer + 1],
                             decay=self.decay[ilayer + 1],
                             spike_fn=self.spike_fn, mode=self.neu_mode),
                    SpikingDropout(self.dropout),
                    LIFReadout(
                        nn.Linear(4096, self.num_class, bias=self.bias),
                        thresh=self.thresh[ilayer + 2],
                        decay=self.decay[ilayer + 2],
                        spike_fn=self.spike_fn, reduction=self.readout, mode=self.neu_mode)
                ]
            elif (vgg_name == 'CIFAR'):
                model = [
                    LIFLayer(nn.Linear(512 * 8 * 8, 1024, bias=self.bias), thresh=self.thresh[ilayer],
                             decay=self.decay[ilayer],
                             spike_fn=self.spike_fn, mode=self.neu_mode),
                    SpikingDropout(self.dropout),
                    LIFLayer(
                        nn.Linear(1024, 512, bias=self.bias), thresh=self.thresh[ilayer + 1],
                        decay=self.decay[ilayer + 1],
                        spike_fn=self.spike_fn, mode=self.neu_mode),
                    SpikingDropout(self.dropout),
                    LIFReadout(nn.Linear(512, self.num_class, bias=self.bias), thresh=self.thresh[ilayer + 2],
                               decay=self.decay[ilayer + 2],
                               spike_fn=self.spike_fn, reduction=self.readout, mode=self.neu_mode)
                ]
            elif (vgg_name == 'ALEX'):
                model = [
                    LIFLayer(nn.Linear(512 * 1 * 1, 1024, bias=self.bias), thresh=self.thresh[ilayer],
                             decay=self.decay[ilayer],
                             spike_fn=self.spike_fn, mode=self.neu_mode),
                    SpikingDropout(self.dropout),
                    LIFLayer(
                        nn.Linear(1024, 1024, bias=self.bias), thresh=self.thresh[ilayer + 1],
                        decay=self.decay[ilayer + 1],
                        spike_fn=self.spike_fn, mode=self.neu_mode),
                    SpikingDropout(self.dropout),
                    LIFReadout(nn.Linear(1024, self.num_class, bias=self.bias), thresh=self.thresh[ilayer + 2],
                               decay=self.decay[ilayer + 2],
                               spike_fn=self.spike_fn, reduction=self.readout, mode=self.neu_mode)
                ]
            elif vgg_name != 'VGG5' and dataset == 'MNIST':
                model = [
                    LIFLayer(nn.Linear(512 * 1 * 1, 4096, bias=self.bias), thresh=self.thresh[ilayer],
                             decay=self.decay[ilayer],
                             spike_fn=self.spike_fn, mode=self.neu_mode),
                    SpikingDropout(self.dropout),
                    LIFLayer(
                        nn.Linear(4096, 4096, bias=self.bias), thresh=self.thresh[ilayer + 1],
                        decay=self.decay[ilayer + 1],
                        spike_fn=self.spike_fn, mode=self.neu_mode),
                    SpikingDropout(self.dropout),
                    LIFReadout(nn.Linear(4096, self.num_class, bias=self.bias), thresh=self.thresh[ilayer + 2],
                               decay=self.decay[ilayer + 2],
                               spike_fn=self.spike_fn, reduction=self.readout, mode=self.neu_mode)
                ]
            elif vgg_name != 'VGG5' and dataset != 'MNIST':
                model = [
                    LIFLayer(nn.Linear(512 * 2 * 2, 4096, bias=self.bias), thresh=self.thresh[ilayer],
                             decay=self.decay[ilayer],
                             spike_fn=self.spike_fn, mode=self.neu_mode),
                    SpikingDropout(self.dropout),
                    LIFLayer(nn.Linear(4096, 4096, bias=self.bias), thresh=self.thresh[ilayer + 1],
                             decay=self.decay[ilayer + 1],
                             spike_fn=self.spike_fn, mode=self.neu_mode),
                    SpikingDropout(self.dropout),
                    LIFReadout(nn.Linear(4096, self.num_class, bias=self.bias), thresh=self.thresh[ilayer + 2],
                               decay=self.decay[ilayer + 2],
                               spike_fn=self.spike_fn, reduction=self.readout, mode=self.neu_mode)
                ]

            if self.hook:
                hook_model = []
                for m in model:
                    hook_model.append(m)
                    if isinstance(m, LIFLayer):
                        hook_model.append(Hook(self.record))
                return nn.Sequential(*hook_model)
            return nn.Sequential(*model)
        else:
            # share weights of classifier with ann
            # if vgg_name != 'VGG5' and dataset != 'MNIST' or (vgg_name == 'VGG5' and dataset != 'MNIST'):
            layers = []
            for i, m in enumerate(ann.classifier):
                if isinstance(m, nn.Linear):
                    # if (i == len(ann.classifier) - 1):
                    #     layers += [
                    #         LIFReadout(m, thresh=self.thresh[ilayer], decay=self.decay[ilayer],
                    #                    spike_fn=self.spike_fn)
                    #     ]
                    # else:
                    layers += [
                        LIFLayer(m, thresh=self.thresh[ilayer], decay=self.decay[ilayer],
                                 spike_fn=self.spike_fn, mode=self.neu_mode)
                    ]
                    if (self.hook):
                        layers += [Hook(self.record)]  # add hook
                    ilayer += 1
                elif isinstance(m, nn.AvgPool2d):
                    layers += [SpikingAvgPool(kernel_size=m.kernel_size, stride=m.stride)]
                    # if (self.hook):
                    #     layers += [Hook(self.record)]  # add hook
                    # ilayer += 1
                    # if (self.hook):
                    #     layers += [Hook(self.record)]  # add hook
                elif isinstance(m, nn.Dropout):
                    layers += [SpikingDropout(m.p)]
                # elif isinstance(m, Hook):
                #     layers += [Hook(self.record)]
            idx = len(layers) - 1
            while (not hasattr(layers[idx], 'func')):
                idx -= 1
            layers[idx] = LIFReadout(layers[idx].func, thresh=layers[idx].thresh, decay=layers[idx].decay,
                                     spike_fn=self.spike_fn, mode=self.neu_mode, reduction=self.readout)
            return nn.Sequential(*layers)

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, SpikingDropout):
                m.reset()
        self.record.clear()
        out = self.features(x)
        # out shape:[T, batch_size, in_channel, H, W ]
        # print('expanding shape:', out.shape)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = self.classifier(out)
        return out

    def _initialize_params(self):
        cfg_name = self.vgg_name
        # prevent overflow in hybrid hook mode
        reserved_num = 10
        if (self.thresh is None):
            self.thresh = [None for i in range(len(cfg[cfg_name]) + reserved_num)]
            # torch.nn.Parameter(torch.empty(len(cfg[cfg_name]) + reserved_num), requires_grad=True)
            # torch.nn.init.normal_(self.thresh, mean=1.0, std=0.01)
        if (self.decay is None):
            self.decay = [None for i in range(len(cfg[cfg_name]) + reserved_num)]
            # self.decay = torch.nn.Parameter(torch.empty(len(cfg[cfg_name]) + reserved_num), requires_grad=True)
            # torch.nn.init.normal_(self.decay, mean=0.9, std=0.01)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 0.01)
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def clamp(self):
        for m in self.modules():
            if isinstance(m, LIFLayer) or isinstance(m, LIFReadout):
                m.clamp()

    def built_from(self, ann):
        self.features = self._make_blocks(ann)
        self.classifier = self._make_fcs(ann)
