"""
-*- coding: utf-8 -*-

@Time    : 2021/5/28 10:29

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : transform.py
"""

import numpy as np
import scipy.io.wavfile as wav
import librosa
import resampy
import torch


class Resample:
    def __init__(self, fs_rsp):
        self.fs_rsp = fs_rsp

    def __call__(self, data):
        wav, fs = data
        return resampy.resample(wav, fs, self.fs_rsp)


class MelSpectrogram:

    def __init__(self, sr, n_fft, hop_length, n_mels, fmin, fmax, delta_order=None, stack=True):
        """
        :param sr,n_fft,hop_length,n_mels, fmin, fmax 参见 librosa.feature.melspectrogram
        :param delta_order: 最高差分阶数，闭区间 [0,delta_order]
        :param stack: stack = True, 返回 [0,delta_order] 阶差分结果, stack = False, 返回 [delta_order,delta_order] 阶差分结果
        """

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.delta_order = delta_order
        self.stack = stack

    def __call__(self, wav):
        # melspectrogram
        S = librosa.feature.melspectrogram(wav,
                                           sr=self.sr,
                                           n_fft=self.n_fft,
                                           hop_length=self.hop_length,
                                           n_mels=self.n_mels,
                                           fmax=self.fmax,
                                           fmin=self.fmin)
        # normalize factor
        M = np.max(np.abs(S))
        if M > 0:
            # log scale
            feat = np.log1p(S / M)
        else:
            feat = S

        if self.delta_order is not None and not self.stack:
            feat = librosa.feature.delta(feat, order=self.delta_order)
            return np.expand_dims(feat.T, 0)

        elif self.delta_order is not None and self.stack:

            feat_list = [feat.T]
            for k in range(1, self.delta_order + 1):
                feat_list.append(librosa.feature.delta(feat, order=k).T)
            return np.stack(feat_list)

        else:
            return np.expand_dims(feat.T, 0)


class Pad:

    def __init__(self, size):
        self.size = size

    def __call__(self, wav):
        wav_size = wav.shape[0]
        # print(wav_size)
        assert self.size >= wav_size, 'padding size must be greater than wav size'
        pad_size = (self.size - wav_size) // 2
        padded_wav = np.pad(wav, ((pad_size, self.size - wav_size - pad_size),), 'constant', constant_values=(0, 0))
        return padded_wav


class Rescale:

    def __call__(self, input):
        """
        :param input: delta_order x T x N
        :return: normalization on time
        """
        std = np.std(input, axis=1, keepdims=True)
        std[std == 0] = 1

        return input / std


class AddNoise:  # Only use for sounds
    def __init__(self, noise=None, path=None, fs=None, fs_rsp=None, SNR=99):
        assert noise or path, "noise content or path should be provided at least one!"
        if not (path is None):
            fs, self.noise = wav.read(path)
        else:
            self.noise = noise
        self.noise = resampy.resample(self.noise, fs, fs_rsp)
        self.fs = fs_rsp
        self.SNR = SNR

    def __call__(self, wav):
        signal = wav / 1.0
        noise = self.noise / 1.0
        # 切割噪音
        assert noise.size >= signal.size, '噪音信号长度应该不短于信号长度'
        beg = np.random.randint(0, noise.size - signal.size + 1)
        noise_split = noise[beg:beg + signal.size]
        # 根据SNR计算加噪后的信息
        P_signal = np.mean(np.power(signal, 2))
        P_noise = np.mean(np.power(noise_split, 2))
        coeff = np.sqrt(P_signal / P_noise / (10 ** (self.SNR / 10)))
        return signal + coeff * noise_split


class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return 'Adding Gaussian Noise for Image Data: ' + '(mean={0}, std={1})'.format(self.mean, self.std)


class TemporalBernolliSpike:
    def __init__(self, rank):
        self.rank = rank

    def __call__(self, spec):
        """
        :param spec: nChannel x T1 x nNeu
        :return: sptn: nChannel x T2 x nNeu
        """
        # normalization
        normal_spec = np.zeros_like(spec)
        num_channel = normal_spec.shape[0]
        for i in range(num_channel):
            spec_min = spec[i, :, :].min()
            spec_max = spec[i, :, :].max()
            normal_spec[i, :, :] = (spec[i, :, :] - spec_min) / (spec_max - spec_min)
        # bernolli spiking
        sptn = np.zeros([num_channel, normal_spec.shape[1] * self.rank, normal_spec.shape[2]])
        base = np.arange(0, sptn.shape[1], self.rank)
        for t in range(self.rank):
            pos = t + base
            sptn[:, pos, :] = (np.random.random(normal_spec.shape) < normal_spec)
        return sptn


class RateCoding:
    def __init__(self, nb_steps, method=1):
        self.nb_steps = nb_steps
        self.method = method

    def __call__(self, img):
        if (self.method != 0):
            assert torch.logical_and(img >= 0, img <= 1).all(), 'input pixel intensity should be normalized into [0,1]'
        spikes = None
        if (self.method == 0):
            spikes = [img.clone() for t in range(self.nb_steps)]
        elif (self.method == 1):
            spikes = [img < torch.rand(*img.shape) for t in range(self.nb_steps)]
        elif (self.method == 2):
            eps = 1e-8
            num_spikes = torch.round(self.nb_steps * img)
            interval = torch.floor(self.nb_steps / (num_spikes + eps))
            spikes = [torch.logical_and(torch.fmod(torch.ones_like(img) * t, interval) == interval - 1,
                                        t / interval < num_spikes)
                      for t in
                      range(self.nb_steps)]
        spikes = torch.stack(spikes)
        return spikes
