"""
-*- coding: utf-8 -*-

@Time    : 2021-11-15 17:51

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : snn.py
"""
import torch
import torch.nn as nn
from model.snn import LIFCell, AIFCell
from util.fold_bn import StraightThrough
from util.util import replaceAct
from model.surrogate_act import SurrogateHeaviside, RectSurrogate


class ANNLayer(nn.Module):
    def __init__(self, fwd_fn, act_fn=nn.ReLU):
        super(ANNLayer, self).__init__()
        self.fwd_fn = fwd_fn
        self.act_fn = act_fn

    def forward(self, x):
        return self.act_fn(self.fwd_fn(x))


class IsomorphicSNN(nn.Module):
    def __init__(self, model, nb_steps, decay, thresh, spike_fn=SurrogateHeaviside.apply, enable_shift=False,
                 enable_init_volt=False,
                 mode='spike', record_interval=None, specials: dict = {}, cell=LIFCell):
        super(IsomorphicSNN, self).__init__()
        self.model = model
        self.nb_steps = nb_steps
        self.spike_fn = spike_fn
        self.decay = decay
        self.thresh = thresh
        self.enable_shift = enable_shift
        self.enable_init_volt = enable_init_volt
        self.mode = mode
        self.record_interval = record_interval
        self.specials = specials
        self.cell = cell
        self.buildup()

    def buildup(self):
        self.composeMoule(self.model)
        self.replaceSNN(self.model)

    def replaceSNN(self, root_module, layer_idx=0):
        layer_idx = layer_idx
        for name, child_moule in root_module.named_children():
            if type(child_moule) in self.specials:
                special_class, special_func = self.specials[type(child_moule)]
                setattr(root_module, name, special_class(child_moule, **special_func(self, layer_idx, child_moule)))
                layer_idx += special_class.num_layer
            if isinstance(child_moule, ANNLayer):
                thresh = self.thresh if isinstance(self.thresh, float) or (
                        isinstance(self.thresh, torch.Tensor) and self.thresh.numel() == 1) else self.thresh[
                    layer_idx]
                decay = self.decay if isinstance(self.decay, float) or (
                        isinstance(self.thresh, torch.Tensor) and self.decay.numel() == 1) else self.decay[
                    layer_idx]
                mem_init = abs(thresh) / 2 if self.enable_init_volt else 0
                shift = abs(thresh) / (2 * self.nb_steps) if self.enable_shift else 0
                setattr(root_module, name,
                        self.cell(child_moule.fwd_fn, self.spike_fn, decay=decay, thresh=thresh, mode=self.mode,
                                  shift=shift, init_mem=mem_init))
                layer_idx += 1
            else:
                layer_idx = self.replaceSNN(child_moule, layer_idx)
        return layer_idx

    def composeMoule(self, root_module, prev_module=None):
        prev_module = prev_module
        for name, child_moule in root_module.named_children():
            if name == 'act_fun' or type(child_moule) in self.specials: continue
            if isinstance(child_moule, nn.Conv2d) or isinstance(child_moule, nn.Linear):
                prev_module = (root_module, name)
                # setattr(root_module, name, ANNLayer(child_moule))
                # prev_module = getattr(root_module, name)
            elif type(child_moule) == type(self.model.act_fun):
                assert prev_module is not None, "there isn't a forward layer before activation function"
                setattr(root_module, name, ANNLayer(fwd_fn=getattr(*prev_module), act_fn=child_moule))
                setattr(prev_module[0], prev_module[1], StraightThrough())
                prev_module = None
            else:
                prev_module = self.composeMoule(child_moule, prev_module=prev_module)
        return prev_module

    def reset_membrane_potential(self):
        for m in self.model.modules():
            if isinstance(m, (LIFCell, AIFCell)):
                m.reset_membrane_potential()

    def forward(self, x):
        self.reset_membrane_potential()
        out = []
        for t in range(self.nb_steps):
            out.append(self.model(x[t]))
        out = torch.stack(out)
        if self.training or self.record_interval is None:
            return out.mean(axis=0)
        else:
            out_dict = {}
            t = 1
            while t <= self.nb_steps:
                # for t in range(self.record_interval, self.nb_steps + 1, self.record_interval):
                out_dict[t] = (out[:t].mean(axis=0))
                t *= self.record_interval
            return out_dict
