"""
-*- coding: utf-8 -*-

@Time    : 2021-11-05 16:14

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : balance_weight_thresh.py
"""
import torch
import torch.nn as nn
from util.hooks import MaxActHook


def getMaxAct(net, ldr, hook_class, device, dtype, end=True, **kwargs):
    """
    Get the maximum activations layer by layer on dataloader ldr
    end = True: record the last fc/conv layer output
    """
    # register hook for special module
    max_act = []
    hooks = []
    names = []
    last_module = None
    for name, module in net.named_modules():
        if name == 'act_fun':  # skip the template variable
            continue
        if isinstance(module, hook_class):
            hooks.append(module.register_forward_hook(MaxActHook(record=max_act, **kwargs)))
            names.append(name)
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            last_module = module
    if end:
        hooks.append(last_module.register_forward_hook(MaxActHook(record=max_act, **kwargs)))
    net.to(device, dtype)
    with torch.no_grad():
        for idx, (ptns, labels) in enumerate(ldr):
            net.eval()
            ptns = ptns.to(device, dtype)
            net(ptns)
    # remove hooks
    for hook in hooks:
        hook.remove()
    return names, max_act


def getMaxActI(net, ldr, device, dtype):
    """
    Get the maximum activations layer by layer on dataloader ldr with customized hook other than pytorch
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


def balanceWeight(net, layer_names, factors, mode='layer_wise'):
    # todo: add supports for channel-wise and  neuron-wise conversion
    if mode != 'layer_wise':
        raise NotImplementedError('neuron_wise and channel_wise are not supported now')
    idx = 0
    prev = None
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prev = module
        if idx < len(layer_names) and name == layer_names[idx] and prev is not None:
            tfactor = factors[idx - 1] if idx > 0 else 1.0
            prev.weight.data *= tfactor / factors[idx]
            if prev.bias is not None:
                prev.bias.data /= factors[idx]
            idx += 1


def balanceThresh():
    """
    For spike rate-based conversion, it's important to modify both thresh and weight
    For PSP or membrane voltage-based conversion, we just need to modify the thresh
    :return:
    """
    pass


def shiftBias(net, shift):
    assert isinstance(shift, float) or isinstance(shift, list), 'shift option must be list or float'
    idx = 0
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.bias.data += shift[idx] if isinstance(shift, list) else shift
            idx += 1
