"""
-*- coding: utf-8 -*-

@Time    : 2021/5/4 11:05

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : util.py
"""

import torch
import numpy as np
import random
import torch.nn as nn
import sys
import copy
import os
from enum import Enum


class Logger(object):
    def __init__(self, path, exist_check=True):
        self.terminal = sys.stdout
        self.log_path = path
        if exist_check:
            # assert not os.path.exists(path), 'the logging file exists already!'
            if os.path.exists(path):
                print('the logging file exists already!')

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, 'a') as f:
            f.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


# redirect output
# sys.stdout = Logger('./Output/log.txt')
# sys.stderr = Logger('./Output/log.txt')
def setup_seed(seed):
    print('set random seed as ' + str(seed))
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def quantile(tensor: torch.Tensor, p: float):
    try:
        return torch.quantile(tensor, p)
    except:
        tensor_np = tensor.cpu().detach().numpy()
        return torch.tensor(np.percentile(tensor_np, q=p * 100)).type_as(tensor)


def transfer_model(src, dst, reshape=False, sequential=True, log=False, model=True, only_weight=False):
    if log:
        print('transferring model weight...')
        print('initial weights:  \n', dst.state_dict())
    src_dict = src
    dst_dict = dst
    if (model):
        src_dict = src.state_dict()
        dst_dict = dst.state_dict()
    saved_dict = {}
    if only_weight:
        for key in list(src_dict.keys()):
            if not key.endswith('bias') and not key.endswith('weight'):
                del src_dict[key]
        for key in list(dst_dict.keys()):
            if not key.endswith('bias') and not key.endswith('weight'):
                del dst_dict[key]
    assert len(src_dict) == len(dst_dict), 'the num of the trainable weights must be same in sequential mode'

    def warp(v):
        if v.requires_grad:
            return nn.Parameter(v.data)
        else:
            return v.data

    for i, (k, v) in enumerate(src_dict.items()):
        if sequential:
            dst_key = list(dst_dict.keys())
            v_ = dst_dict[dst_key[i]]
            if (v_.numel() == v.numel() == 1 or v.shape == v_.shape):
                saved_dict[dst_key[i]] = warp(v)
            elif (reshape):
                saved_dict[dst_key[i]] = torch.reshape(warp(v), v_.shape)
            else:
                raise NotImplementedError(
                    'Error: source model with shape {} and destination model with shape {} are mismatched. '.format(
                        src_dict[k].shape, dst_dict[k].shape))
        else:
            if k in dst_dict.keys():
                if (src_dict[k].shape == dst_dict[k].shape):
                    saved_dict[k] = warp(src_dict[k])
                elif (reshape):
                    saved_dict[k] = torch.reshape(warp(src_dict[k]), dst_dict[k].shape)
                else:
                    raise NotImplementedError(
                        'Error: source model with shape {} and destination model with shape {} are mismatched. '.format(
                            src_dict[k].shape, dst_dict[k].shape))
    if model:
        dst.load_state_dict(saved_dict, strict=False)
    else:
        return saved_dict
    if log:
        print('weights after transferring:  \n', dst.state_dict())


def replaceAct(net, src_act_type, dst_acts, rename_act=True):
    """
    :param net:  the network needed to replace activation_function
    :param src_act_type: the type of source activation functions needed to be replaced with
    :param dst_acts: the target activation need to be replace
                     if isinstance(dst_acts,list), bfs the net and replace src_act with dst_acts[i] one by one
                     else isinstance(dst_acts,nn.Module), replace all the source activation functions with dst_acts
    :return:
    """
    assert isinstance(dst_acts, list) or isinstance(dst_acts,
                                                    nn.Module), 'Destination activation function should be list or nn.Module'
    if rename_act:
        dst_act = dst_acts[0] if isinstance(dst_acts, list) else dst_acts
        setattr(net, 'act_fun', copy.deepcopy(dst_act))
    with torch.no_grad():
        queue = [net]
        idx = 0
        while len(queue) > 0:
            module = queue.pop()
            for name, child in module.named_children():
                if name == 'act_fun': continue  # skip the template
                if isinstance(child, src_act_type):
                    dst_act = (dst_acts[idx] if isinstance(dst_acts, list) else copy.deepcopy(dst_acts))
                    setattr(module, name, dst_act)
                    idx += 1
                else:
                    queue.insert(0, child)


def fetchActs(net, src_act_type):
    acts = []
    with torch.no_grad():
        queue = [net]
        idx = 0
        while len(queue) > 0:
            module = queue.pop()
            for name, child in module.named_children():
                if name == 'act_fun': continue  # skip the template
                if isinstance(child, src_act_type):
                    acts.append(child)
                else:
                    queue.insert(0, child)
    return acts
    # i = 0
    # for l, m in enumerate(net.features.children()):
    #     if isinstance(m, src_act):
    #         net.features[l] = dst_acts[i]
    #         i += 1
    # for l, m in enumerate(net.classifier.children()):
    #     if isinstance(m, src_act):
    #         net.classifier[l] = dst_acts[i]
    #         i += 1
    # return net


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
