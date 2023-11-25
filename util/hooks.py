"""
-*- coding: utf-8 -*-

@Time    : 2021-10-02 10:07

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : hooks.py
"""
import torch
import torch.nn as nn
from util.util import quantile


class Hook(nn.Module):
    # self-implemented Hook as a flag to adjust model and obtain intern activation
    # just record the output in variable 'record'
    def __init__(self, record, log=False):
        super(Hook, self).__init__()
        self.log = log
        self.record = record

    def forward(self, x):
        # Do your print / debug stuff here
        self.record.append(x)
        if (self.log):
            print(x)
        return x


class RecordHook:
    def __init__(self, to_cpu=False):
        self.inputs = []
        self.extern_inputs = []
        self.outputs = []
        self.to_cpu = to_cpu

    def __call__(self, module, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        device = output.device if not self.to_cpu else 'cpu'
        self.outputs.append(output.data.clone().to(device))
        self.inputs.append(input[0].data.clone().to(device))
        if len(input) == 1:
            self.extern_inputs.append(torch.zeros_like(output).to(device))
        elif len(input) == 2:
            self.extern_inputs.append(input[1].data.clone().to(device))
        else:
            raise NotImplementedError('not support for packed inputs with size > 2 now')

    def clear(self):
        # del self.inputs
        # del self.outputs
        # del self.extern_inputs
        self.inputs = []
        self.outputs = []
        self.extern_inputs = []

    def get(self, idx):
        assert idx < len(self.inputs), 'the index is greater than the maximum cache size'
        return self.inputs[idx], self.outputs[idx], self.extern_inputs[idx]

    def reset(self):
        inputs = torch.stack(self.inputs)
        outputs = torch.stack(self.outputs)
        extern_inputs = torch.stack(self.extern_inputs)
        self.clear()

        return inputs, outputs, extern_inputs


class SumHook:
    def __init__(self, to_cpu=False):
        self.inputs = 0
        self.extern_inputs = 0
        self.outputs = 0
        self.to_cpu = to_cpu

    def __call__(self, module, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        device = output.device if not self.to_cpu else 'cpu'
        self.outputs += (output.data.clone().to(device))
        self.inputs += (input[0].data.clone().to(device))
        if len(input) == 1:
            self.extern_inputs += (torch.zeros_like(output).to(device))
        elif len(input) == 2:
            self.extern_inputs += (input[1].data.clone().to(device))
        else:
            raise NotImplementedError('not support for packed inputs with size > 2 now')


class DPSumHook:
    def __init__(self, to_cpu=False):
        self.inputs = 0
        self.extern_inputs = 0
        self.outputs = 0
        self.to_cpu = to_cpu
        self.gpu_inputs = [0 for i in range(torch.cuda.device_count())]
        self.gpu_outputs = [0 for i in range(torch.cuda.device_count())]
        self.gpu_extern_inputs = [0 for i in range(torch.cuda.device_count())]

    def __call__(self, module, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        device = output.device if not self.to_cpu else 'cpu'
        self.gpu_outputs[output.get_device()] += output.to(device)
        self.gpu_inputs[input[0].get_device()] += input[0].to(device)
        self.gpu_extern_inputs[input[0].get_device()] += torch.zeros_like(input[0]).to(device)
        if len(input) == 2:
            self.gpu_extern_inputs[input[1].get_device()] += input[1].to(device)

    def msync(self):
        inputs, extern_inputs, outputs = torch.cat(self.gpu_inputs), torch.cat(self.gpu_extern_inputs), torch.cat(
            self.gpu_outputs)
        self.gpu_inputs = [0 for i in range(torch.cuda.device_count())]
        self.gpu_outputs = [0 for i in range(torch.cuda.device_count())]
        self.gpu_extern_inputs = [0 for i in range(torch.cuda.device_count())]
        return inputs, extern_inputs, outputs


class DPRecordHook:
    def __init__(self, to_cpu=False):
        self.to_cpu = to_cpu
        self.gpu_inputs = [[] for i in range(torch.cuda.device_count())]
        # self.gpu_outputs = [[] for i in range(torch.cuda.device_count())]
        self.gpu_extern_inputs = [[] for i in range(torch.cuda.device_count())]
        self.ext_in = False

    def __call__(self, module, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        device = output.device if not self.to_cpu else 'cpu'
        # self.gpu_outputs[output.get_device()].append(output.to(device))
        self.gpu_inputs[input[0].get_device()].append(input[0].to(device))
        if len(input) == 2:
            self.ext_in = True
            self.gpu_extern_inputs[input[1].get_device()].append(input[1].to(device))
        else:
            self.gpu_extern_inputs[input[0].get_device()].append(torch.Tensor([0]))

    def msync(self):
        for i, inputs in enumerate(self.gpu_inputs):
            self.gpu_inputs[i] = torch.stack(inputs)
        for i, inputs in enumerate(self.gpu_extern_inputs):
            self.gpu_extern_inputs[i] = torch.stack(inputs)
        # for i, inputs in enumerate(self.gpu_outputs):
        #     self.gpu_outputs[i] = torch.stack(inputs)
        inputs = torch.cat(self.gpu_inputs, axis=1)
        # extern_inputs = torch.cat(self.gpu_extern_inputs,
        #                           axis=1)
        if self.ext_in:
            extern_inputs = torch.cat(self.gpu_extern_inputs,
                                      axis=1)
        else:
            extern_inputs = self.gpu_extern_inputs[0]
        # inputs, extern_inputs, outputs = torch.cat(self.gpu_inputs, axis=1), torch.cat(self.gpu_extern_inputs,
        #                                                                                axis=1), torch.cat(
        #     self.gpu_outputs, axis=1)
        self.gpu_inputs = [[] for i in range(torch.cuda.device_count())]
        # self.gpu_outputs = [[] for i in range(torch.cuda.device_count())]
        self.gpu_extern_inputs = [[] for i in range(torch.cuda.device_count())]
        return inputs, extern_inputs


class MaxActHook:
    # todo: implement those perserved parts
    #  fetch the maximum value with pytorch hook mechanism
    def __init__(self, record, scale_mode='layer_wise', percentile=None, momentum=None, spike=False, abs=True):
        self.momentum = momentum
        self.percentile = percentile
        self.spike = spike
        self.scale_mode = scale_mode
        self.running_max = 0
        self.abs = abs
        # record running_max into external list
        self.record = record
        self.layer_idx = len(self.record)
        self.record.append(self.running_max)
        self.__inspection()

    def __inspection(self):
        if self.scale_mode not in ['channel_wise', 'neuron_wise', 'layer_wise']:
            raise NotImplementedError("Only mode 'channel_wise', 'neuron_wise' and 'layer_wise' are in the plan")
        assert self.momentum is None or 0 <= self.momentum <= 1, 'specified momentum should be in the range [0,1].'
        assert self.percentile is None or 0 <= self.percentile <= 1, 'specified percentile should be in the range [0,1].'

    def __call__(self, module, input, output):
        """
        :param module:
        :param input: if spike = False, the shape should be batch_size x channel x h x w
                      if spike = True, the shape should be T x batch_size x channel x h x w
        :param output: the same with input
        """
        if self.spike:
            output = output.detach().clone().mean(axis=0)
        if self.abs:
            output = output.abs()
            # raise NotImplementedError('the mode for maximum spike rate is not supported now')
        max_act = None
        if self.scale_mode == 'layer_wise':
            if self.percentile:
                max_act = quantile(output, self.percentile)
            else:
                max_act = output.max()
        if self.momentum is not None:
            self.running_max = self.momentum * self.running_max + (1 - self.momentum) * max_act
        else:
            self.running_max = max(self.running_max, max_act.detach().item())
        self.record[self.layer_idx] = self.running_max
