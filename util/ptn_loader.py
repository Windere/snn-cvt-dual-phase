"""
-*- coding: utf-8 -*-

@Time    : 2021-04-13 10:47

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : ptn_loader.py
"""

import torch
import numpy as np
import torch.nn as nn
import os
import soundfile
from torch.utils.data import DataLoader, Dataset


class audioDataset(Dataset):
    def __init__(self, audio_list, label2idx, extract_label, transform=None):
        self.audio_list = audio_list
        self.transform = transform
        self.extract_label = extract_label
        self.label2idx = label2idx

    def __getitem__(self, index):
        with soundfile.SoundFile(self.audio_list[index]) as f:
            f.seek(0)
            wav = f.read()
            if self.transform:
                wav = self.transform([wav, f.samplerate])
            label_name = self.extract_label(os.path.split(self.audio_list[index])[1])
            return wav, self.label2idx[label_name]

    def __len__(self):
        return len(self.audio_list)


# todo: 复习python异常处理机制
# todo: 实现 neuron-based 和 time-based 的Spiking Dataset
class sPtn(Dataset):
    def __init__(self, dataSource, labelSource, TmaxSource, mode='spike'):
        self.mode = mode
        if mode == 'spike':
            """
            dataSource: num_samples x max_num_spikes x 2, pad with -1 for inconsistency in spiking counts.
                        eg:
                            num_samples = 2 
                            max_num_spikes = 5
                            sample 1:
                                for neuron 1 , the afferent is [2, 5, 7]
                                for neuron 2 , the afferent is [1, 9]
                            sample 2: 
                                for neuron 1 : the afferent is [2, 9, 14]
                                for neuron 2 , the afferent is [6]
                            ==============================================================================
                            then dataSource = [
                                                [[1,2],[1,5],[1,7],[2,1],[2,9]], # for sample 1 
                                                [[1,2],[1,9],[1,14],[2,6],[-1,-1]] # for sample 2
                                                                ]]
            """
            self.dataSource = dataSource
            self.labelSource = labelSource
            self.TmaxSource = TmaxSource
        elif mode == 'neuron':
            raise NotImplementedError('Neuron-based loading is not supported util now.')
        elif mode == 'time':
            raise NotImplementedError('Time-based loading is not supported util now.')
        else:
            raise NotImplementedError("Only 'spike' ,'neuron' and 'time' format are in the plan.")

    def __getitem__(self, index):
        if self.mode == 'spike':
            spike = self.dataSource[index, :, :]
            spike = np.transpose(spike)
            label = self.labelSource[index]
            Tmax = float(self.TmaxSource[index])
            return spike, Tmax, label

    def __len__(self):
        return self.dataSource.shape[0]


def transeForm(ptn, num_neuron, num_time):
    """"
    :param ptn: batch_size*2*num_spike torch.tensor { ptn[batch_idx][0][spike_idx] = neuron_idx, ptn[batch_idx][1][spike_idx] = time_idx }
    :return:   batch_size*num_neurons*num_time  torch.tensor
    """
    batch_size = ptn.shape[0]
    batch_idx = torch.arange(0, batch_size).reshape([batch_size, 1, 1])
    index = torch.cat([batch_idx.repeat(1, 1, ptn.shape[2]), ptn.long()], dim=1).permute([0, 2, 1]).reshape(-1,
                                                                                                            3).long()
    # print(index.shape)
    index = index[index[:, 1] > 0, :]
    index = (index[:, 0], index[:, 1] - 1, index[:, 2])
    new_ptn = torch.zeros([batch_size, num_neuron, num_time])
    new_ptn[index] = 1
    return new_ptn


# todo: 封装为类
def collate_func(mode, num_in):
    """
    :param mode: 脉冲模式的组织形式
    :param num_in: 输入神经元数目
    :return: 返回指定num_in输入神经元数量的collate_func(Data)
    """
    if mode == 'spike':
        num_neurons = num_in

        def _spike_collate_func(data):
            X_batch = torch.tensor([d[0] for d in data])
            tmax = torch.tensor([d[1] for d in data])
            y_batch = torch.tensor([d[2] for d in data]).to(torch.int64)
            # num_neurons = X_batch[:, 0, :].max() + 1
            X_batch = transeForm(X_batch, num_neurons, int(tmax.max().item() + 1))
            return X_batch, y_batch

        return _spike_collate_func
    else:
        raise NotImplementedError("Only 'spike' ,'neuron' and 'time' format are in the plan.")


def spike_collate_func(data):
    X_batch = torch.tensor([d[0] for d in data])
    tmax = torch.tensor([d[1] for d in data])
    y_batch = torch.tensor([d[2] for d in data]).to(torch.int64)
    num_neurons = X_batch[:, 0, :].max() + 1
    X_batch = transeForm(X_batch, num_neurons, int(tmax.max().item() + 1))
    return X_batch, y_batch
