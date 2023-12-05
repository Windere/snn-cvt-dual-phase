"""
-*- coding: utf-8 -*-

@Time    : 2021/4/26 15:18

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : snn.py

Reference:
    1. Gu, Pengjie, et al. "STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep Spiking Neural Networks." IJCAI. 2019.
    2. Zimmer, Romain, et al. "Technical report: supervised training of convolutional spiking neural networks with PyTorch." arXiv preprint arXiv:1911.10124 (2019).
    3. 加速自定义RNN Cell: https://github.com/pytorch/pytorch/blob/963f7629b591dc9750476faf1513bc7f1fb4d6de/benchmarks/fastrnns/custom_lstms.py#L246
"""
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class SNN(torch.nn.Module):

    def __init__(self, layers):

        super(SNN, self).__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        loss_seq = []
        #  每一层的loss是什么?（Spike Rate Loss）  loss = 0.5*(spk_rec**2).mean()
        for l in self.layers:
            if isinstance(l, tdBatchNorm):
                x = x.unsqueeze(1)
                x = x.unsqueeze(-1)
                # print(x.shape)
                x = l(x)
                x = x.squeeze()
            else:
                x, loss = l(x)
                loss_seq.append(loss)

        return x, loss_seq

    def clamp(self):

        for l in self.layers:
            if not isinstance(l, tdBatchNorm):
                l.clamp()

    def reset_parameters(self):

        for l in self.layers:
            if not isinstance(l, tdBatchNorm):
                l.reset_parameters()


class STCADenseLayer(torch.nn.Module):

    def __init__(self, input_shape, output_shape, spike_fn, w_init_mean, w_init_std,
                 recurrent=False, lateral_connections=True, rc_drop=-1, fc_drop=-1, eps=1e-8):

        super(STCADenseLayer, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.spike_fn = spike_fn
        self.recurrent = recurrent
        self.eps = eps
        self.lateral_connections = lateral_connections

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.w = torch.nn.Parameter(torch.empty((input_shape, output_shape)), requires_grad=True)
        if recurrent:
            self.v = torch.nn.Parameter(torch.empty((output_shape, output_shape)), requires_grad=True)

        # RNN Dropout
        self.fc_drop = fc_drop
        self.rc_drop = rc_drop
        if fc_drop > 0:
            self.drop_fc = td_Dropout(fc_drop, True)
        if rc_drop > 0:
            self.drop_rc = td_Dropout(rc_drop, True)

        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)  # threshhold

        # decay
        self.decay_m = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.decay_s = torch.nn.Parameter(torch.empty(1), requires_grad=True)
        self.decay_e = torch.nn.Parameter(torch.empty(1), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        self.spk_rec_hist = None

        self.training = True

    def forward(self, x):

        batch_size = x.shape[0]
        # update drop out mask
        if self.fc_drop > 0:
            self.drop_fc.reset()
        if self.rc_drop > 0:
            self.drop_rc.reset()
        # todo: 爱因斯坦求和标记，支持torch BP
        h = torch.einsum("abc,cd->abd", x, self.w)
        nb_steps = h.shape[1]

        # output spikes
        spk = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
        # refractory period kernel
        E = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
        # input response kernel
        M = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)
        S = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)

        # output spikes recording
        spk_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)

        if self.lateral_connections:  # 计算侧抑制权重
            d = torch.einsum("ab, ac -> bc", self.w, self.w)

        norm = (self.w ** 2).sum(0)  # 对每个输出神经元计算norm以约束权重

        for t in range(nb_steps):
            # input term
            input_ = h[:, t, :]
            if self.recurrent:
                if self.rc_drop > 0:
                    input_ = input_ + torch.einsum("ab,bc->ac", self.drop_rc(spk), self.v)
                else:
                    input_ = input_ + torch.einsum("ab,bc->ac", spk, self.v)

            # todo: add reset mechanism
            if self.lateral_connections:
                # 模拟侧抑制
                E = torch.einsum("ab,bc ->ac", spk, d)
            else:
                # 模拟refractory period
                E = self.decay_m * norm * (E + spk)

            M = self.decay_m * (M + input_)
            S = self.decay_s * (S + input_)

            # membrane potential update
            mem = M - S - E
            mthr = torch.einsum("ab,b->ab", mem, 1. / (norm + self.eps)) - self.b

            spk = self.spike_fn(mthr)
            spk_rec[:, t, :] = spk
            if self.fc_drop > 0:
                spk = self.drop_fc(spk)

            # save spk_rec for plotting
        self.spk_rec_hist = spk_rec.detach().cpu().numpy()  # 记录该层的脉冲发放 shape: batch_size x T x NeuIn

        loss = 0.5 * (spk_rec ** 2).mean()

        return spk_rec, loss

    # todo: 将权重的初始化逻辑从module中独立出来形成Initilizer
    def reset_parameters(self):

        torch.nn.init.normal_(self.w, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.input_shape))
        if self.recurrent:
            torch.nn.init.normal_(self.v, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.output_shape))
        torch.nn.init.normal_(self.decay_m, mean=0.6, std=0.01)
        torch.nn.init.normal_(self.decay_s, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.decay_e, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def reset_parameters_v2(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean, std=self.w_init_std * np.sqrt(1. / self.input_shape))
        if self.recurrent:
            torch.nn.init.orthogonal(self.v)
        torch.nn.init.normal_(self.decay_m, mean=0.6, std=0.01)
        torch.nn.init.normal_(self.decay_s, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.decay_e, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def reset_parameters_v3(self):
        torch.nn.init.orthogonal(self.w)
        if self.recurrent:
            torch.nn.init.orthogonal(self.v)
        torch.nn.init.normal_(self.decay_m, mean=0.6, std=0.01)
        torch.nn.init.normal_(self.decay_s, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.decay_e, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def reset_parameters_v4(self):
        torch.nn.init.xavier_normal_(self.w)
        if self.recurrent:
            torch.nn.init.xavier_normal_(self.v)
        torch.nn.init.normal_(self.decay_m, mean=0.6, std=0.01)
        torch.nn.init.normal_(self.decay_s, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.decay_e, mean=0.9, std=0.01)
        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    # todo: 将decay改为可学习的softmax门控机制
    def clamp(self):
        self.decay_m.data.clamp_(0., 1.)
        self.decay_e.data.clamp_(0., 1.)
        self.decay_s.data.clamp_(0., 1.)
        self.b.data.clamp_(min=0.)


class ReadoutLayer(torch.nn.Module):
    "Fully connected readout"

    def __init__(self, input_shape, output_shape, w_init_mean, w_init_std, eps=1e-8, time_reduction="mean"):

        assert time_reduction in ["mean", "max"], 'time_reduction should be "mean" or "max"'

        super(ReadoutLayer, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.w_init_mean = w_init_mean
        self.w_init_std = w_init_std

        self.eps = eps
        self.time_reduction = time_reduction

        self.w = torch.nn.Parameter(torch.empty((input_shape, output_shape)), requires_grad=True)
        if time_reduction == "max":
            self.beta = torch.nn.Parameter(torch.tensor(0.7 * np.ones((1))), requires_grad=True)
        self.b = torch.nn.Parameter(torch.empty(output_shape), requires_grad=True)

        self.reset_parameters()
        self.clamp()

        self.mem_rec_hist = None

    def forward(self, x):

        batch_size = x.shape[0]

        h = torch.einsum("abc,cd->abd", x, self.w)

        norm = (self.w ** 2).sum(0)

        if self.time_reduction == "max":
            nb_steps = x.shape[1]
            # membrane potential
            mem = torch.zeros((batch_size, self.output_shape), dtype=x.dtype, device=x.device)

            # memrane potential recording
            mem_rec = torch.zeros((batch_size, nb_steps, self.output_shape), dtype=x.dtype, device=x.device)

            for t in range(nb_steps):
                # membrane potential update
                mem = mem * self.beta + (1 - self.beta) * h[:, t, :]
                mem_rec[:, t, :] = mem

            output = torch.max(mem_rec, 1)[0] / (norm + 1e-8) - self.b

        elif self.time_reduction == "mean":

            mem_rec = h
            output = torch.mean(mem_rec, 1) / (norm + 1e-8) - self.b

        # save mem_rec for plotting
        self.mem_rec_hist = mem_rec.detach().cpu().numpy()

        loss = None

        return output, loss

    def reset_parameters(self):
        torch.nn.init.normal_(self.w, mean=self.w_init_mean,
                              std=self.w_init_std * np.sqrt(1. / (self.input_shape)))

        if self.time_reduction == "max":
            torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)

        torch.nn.init.normal_(self.b, mean=0., std=0.01)

    def reset_parameters_v4(self):
        torch.nn.init.xavier_normal_(self.w)

        if self.time_reduction == "max":
            torch.nn.init.normal_(self.beta, mean=0.7, std=0.01)

        torch.nn.init.normal_(self.b, mean=1., std=0.01)

    def clamp(self):

        if self.time_reduction == "max":
            self.beta.data.clamp_(0., 1.)


class LIFLayer(torch.nn.Module):
    # todo: Initialize， Trainable decay or constant [finish]
    # todo: split the status updating as a class [finish] ==> Class LIFCell
    # todo: compare the result of ||w||*thresh and thresh

    """"
    the simplest iterative equation of LIF equations
    """

    def __init__(self, func, spike_fn, decay=None, thresh=None, mode='psp', init_mem=None):
        super(LIFLayer, self).__init__()
        self.func = func
        self.decay = decay
        self.thresh = thresh
        self.spike_fn = spike_fn
        self.mode = mode
        self.residual_mem = None
        self.init_mem = init_mem
        self.reset_parameters()
        if (mode == 'psp'):
            print('model "psp" only use for conversion and must guarantee thresh holds same !!! ')

    def forward(self, x):
        """
        Be care about memory here !!!
        :param x: T * batch_size *num_channel * H * W spike pattern
                  or T* batch_size *channel
        :return:
        """
        T = x.shape[0]
        spike, mem = None, None
        spike_rec = []
        for t in range(T):
            current = self.func(x[t])  # input current
            # if self.mode == 'spike':
            #     current += 1 / (2 * T)
            if self.mode == 'psp':
                current += self.thresh / (2 * T)
            if mem is None:
                if self.init_mem is None:
                    mem = torch.zeros_like(current)
                else:
                    mem = self.init_mem.clone()
            mem = self.decay * mem + current
            spike = self.spike_fn(mem - self.thresh)
            # spike_rec.append(spike.clone())
            if self.mode == 'psp':
                spike_rec.append((spike.float() * self.thresh).clone())
            elif self.mode == 'spike':
                spike_rec.append(spike.clone())
            mem -= spike.float() * self.thresh
            # print(mem.max())
        self.residual_mem = mem
        out = torch.stack(spike_rec)
        return out

    def reset_parameters(self):
        if (self.thresh is None):
            self.thresh = torch.nn.Parameter(torch.empty(1), requires_grad=True)
            torch.nn.init.normal_(self.thresh, mean=1.0, std=0.01)

        if (self.decay is None):
            self.decay = torch.nn.Parameter(torch.empty(1), requires_grad=True)
            torch.nn.init.normal_(self.decay, mean=0.9, std=0.01)

    def clamp(self):
        self.decay.data.clamp_(0., 1.)
        self.thresh.data.clamp_(min=0.)


# class LIFCell(torch.nn.Module):
#     #  the status updating as a class
#     """"
#     the simplest iterative equation of LIF equations
#     """
#
#     def __init__(self, func, spike_fn, decay=None, thresh=None, mode='psp', init_mem=0, shift=0):
#         super(LIFCell, self).__init__()
#         self.func = func
#         self.decay = decay
#         self.thresh = thresh
#         self.spike_fn = spike_fn
#         self.mode = mode
#         self.mem = 0
#         self.init_mem = init_mem
#         self.shift = shift
#         self.t = 0
#         self.reset_parameters()
#         if (mode == 'psp'):
#             print('model "psp" only use for conversion and must guarantee thresh holds same !!! ')
#
#     def forward(self, x, extern_current=0):
#         """
#         :param x: batch_size *in_channel * Hi * Wi spike pattern
#                mem: batch_size *out_channel * Ho * Wo
#                or  batch_size *channel for a specified time step
#         :return:
#         """
#         self.t += 1
#         current = self.func(x) + self.shift  # input current
#         if isinstance(self.mem, torch.Tensor):
#             self.mem = self.mem.to(current.device)
#         self.mem = self.decay * self.mem + current + extern_current
#         spike = self.spike_fn(self.mem - self.thresh)
#         self.mem = self.mem - spike * self.thresh
#         return spike
#
#     def reset_membrane_potential(self):
#         self.t = 0
#         self.mem = self.init_mem
#         if not isinstance(self.init_mem, float) and not isinstance(self.init_mem, int):
#             self.mem = self.init_mem.clone()
#
#     def reset_parameters(self):
#         if (self.thresh is None):
#             self.thresh = torch.nn.Parameter(torch.empty(1), requires_grad=True)
#             torch.nn.init.normal_(self.thresh, mean=1.0, std=0.01)
#
#         if (self.decay is None):
#             self.decay = torch.nn.Parameter(torch.empty(1), requires_grad=True)
#             torch.nn.init.normal_(self.decay, mean=0.9, std=0.01)
#
#     def clamp(self):
#         self.decay.data.clamp_(0., 1.)
#         self.thresh.data.clamp_(min=0.)
class LIFCell(torch.nn.Module):
    #  the status updating as a class
    """"
    the simplest iterative equation of LIF equations
    """

    def __init__(self, func, spike_fn, decay=None, thresh=None, mode='psp', init_mem=0, shift=0):
        super(LIFCell, self).__init__()
        self.func = func
        self.decay = decay
        self.thresh = thresh
        self.spike_fn = spike_fn
        self.mode = mode
        self.mem = 0
        self.init_mem = init_mem
        self.shift = shift
        self.t = 0
        self.reset_parameters()
        # if (mode == 'psp'):
        #     print('model "psp" only use for conversion and must guarantee thresh holds same !!! ')

    def forward(self, x, extern_current=0):
        """
        :param x: batch_size *in_channel * Hi * Wi spike pattern
               mem: batch_size *out_channel * Ho * Wo
               or  batch_size *channel for a specified time step
        :return:
        """
        self.t += 1
        # print(self.func.parameters().device)
        current = self.func(x) + self.shift  # input current
        self.mem = self.decay * self.mem + current + extern_current
        spike = self.spike_fn(self.mem - self.thresh)
        self.mem = self.mem - spike * self.thresh
        if self.mode == 'psp':
            spike *= self.thresh
        return spike

    def reset_membrane_potential(self):
        self.t = 0
        self.mem = self.init_mem
        if not isinstance(self.init_mem, float) and not isinstance(self.init_mem, int):
            self.mem = self.init_mem.clone()

    def reset_parameters(self):
        if (self.thresh is None):
            self.thresh = torch.nn.Parameter(torch.empty(1), requires_grad=True)
            torch.nn.init.normal_(self.thresh, mean=1.0, std=0.01)

        if (self.decay is None):
            self.decay = torch.nn.Parameter(torch.empty(1), requires_grad=True)
            torch.nn.init.normal_(self.decay, mean=0.9, std=0.01)

    def clamp(self):
        self.decay.data.clamp_(0., 1.)
        self.thresh.data.clamp_(min=0.)


class LIFReadout(torch.nn.Module):
    """"
    the simplest iterative equation of LIF equations
    """

    def __init__(self, func, decay, thresh, spike_fn, reduction='rate', mode='psp', init_mem=None):
        super(LIFReadout, self).__init__()
        self.func = func
        self.decay = decay
        self.thresh = thresh
        self.spike_fn = spike_fn
        self.reduction = reduction
        self.mode = mode
        self.init_mem = init_mem
        self.reset_parameters()
        if (mode == 'psp'):
            print('model "psp" only use for conversion and must guarantee thresh holds same !!! ')
        # todo: Initialize， Trainable decay or constant [finish]

    def forward(self, x):
        """
        Be care about memory here !!!
        :param x: T * batch_size *num_channel * H * W spike pattern
                  or T* batch_size *channel
        :return:
        """
        T = x.shape[0]
        spike, mem = None, None
        spike_rec = []
        mem_rec = []
        cum_rec = []
        for t in range(T):
            current = self.func(x[t])  # input current
            # if self.mode == 'spike':
            #     current += 1 / (2 * T)
            if self.mode == 'psp':
                current += self.thresh / (2 * T)
            if (mem is None):
                if self.init_mem is None:
                    mem = torch.zeros_like(current)
                else:
                    mem = self.init_mem.clone()
            mem = self.decay * mem + current
            # if self.mode == 'spike':
            #     mem = self.decay * mem + self.thresh * current
            # else:
            #     mem = self.decay * mem + current
            # todo: soft-reset vs. hard reset vs. neither
            # todo: compare the result of ||w||*thresh and thresh
            spike = self.spike_fn(mem - self.thresh)
            mem = mem - spike * self.thresh
            mem_rec.append(mem.clone())
            spike_rec.append(spike.clone())
            cum_rec.append(current.clone())
        spikes = torch.stack(spike_rec)
        mems = torch.stack(mem_rec)
        cum_rec = torch.stack(cum_rec)
        self.residual_mem = mem
        if (self.reduction == 'max'):
            return mems.max(axis=0)[0]
        if (self.reduction == 'mean'):
            return mems.mean(axis=0)
        if (self.reduction == 'rate'):
            return spikes.mean(axis=0)
        if (self.reduction == 'mean_cum'):
            return cum_rec.mean(axis=0)

    def reset_parameters(self):
        if (self.thresh is None):
            self.thresh = torch.nn.Parameter(torch.empty(1), requires_grad=True)
            torch.nn.init.normal_(self.thresh, mean=1.0, std=0.01)
        if (self.decay is None):
            self.decay = torch.nn.Parameter(torch.empty(1), requires_grad=True)
            torch.nn.init.normal_(self.decay, mean=0.9, std=0.01)

    def clamp(self):
        self.decay.data.clamp_(0., 1.)
        self.thresh.data.clamp_(min=0.)


class SpikingAvgPool(torch.nn.Module):
    """
    Data format:  T * batch_size *num_channel * H * W  for contiguous indexing
                  so customized Pooling layer is adopted rather than nn.AvgPool3d with kernel depth = 1
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        super(SpikingAvgPool, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                                 count_include_pad=count_include_pad, divisor_override=divisor_override)

    def forward(self, x):
        T = x.shape[0]
        out = []
        for t in range(T):
            out.append(self.pool(x[t]))
        return torch.stack(out)


class SpikingDropout(nn.Module):
    def __init__(self, p=0.5, dropout_spikes=False):
        super(SpikingDropout, self).__init__()
        assert 0 <= p <= 1
        self.mask = None
        self.p = p
        self.dropout_spikes = dropout_spikes

    def create_mask(self, x: torch.Tensor):
        self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)

    def forward(self, x: torch.Tensor):
        if self.training:
            if self.mask is None:
                self.create_mask(x)
            if self.dropout_spikes:
                return mul(self.mask, x)
            else:
                return x.mul(self.mask) / (1 - self.p)
        else:
            return x

    def reset(self):
        self.mask = None
