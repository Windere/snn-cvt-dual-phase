"""
-*- coding: utf-8 -*-

@Time    : 2021-11-16 19:37

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : main_convert.py
"""
import sys

sys.path.append('.')
sys.path.append("..")
sys.path.append("../..")
# todo: sys.path.append 仅适用于测试, 实际调用应在项目根路径下将测试代码作为模块调用
import re
import csv
import torch
import copy
import math
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from models.vgg import rVGG
from models.LeNet import LeNet5
import torch.optim as optim
from util.transform import RateCoding
from util.util import setup_seed, Logger
from util.util import transfer_model, replaceAct
from util.hooks import RecordHook
from torch.utils.tensorboard import SummaryWriter
from models.resnet import ResNetX
import torch.nn.functional as F
from util.fold_bn import search_fold_and_remove_bn, StraightThrough, search_fold_and_reset_bn
from models.snn import IsomorphicSNN
from model.snn import LIFCell, LIFLayer, AIFCell
from model.surrogate_act import QCS, MovingQCS, TrainableQCS, Clamp, NoisyQCS, TClamp, NTClamp, NoisyQC, LeackyQCS, \
    LeackyClamp
from model.surrogate_act import SurrogateHeaviside, RectSurrogate, NoisySpike
from models.balance_weight_thresh import getMaxAct, balanceWeight, shiftBias
from main_ann import loadData, runTrain, runTest, get_model_name
from util.scheduler import GradualWarmupScheduler
from util.optim import RAdam
import warnings
from models.mobilenet import MobileNetV2
from models.resnetxt import ResNeXt29_2x64d

warnings.filterwarnings("ignore")
results = {'model_name': '', 'ann': '', 'fold_ann': '', 'ann2snn': '', 'qc_ann': '', 'fintune_qc_ann': '',
           'fintune_qc_ann2snn': '',
           'pc_snn': '', 'wc_pc_snn': '', 'pc_wc_pc_snn': '', 'cmd': ''
           }


class EarlyStoping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_state_dict = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        self.best_state_dict = copy.deepcopy(model.state_dict())


def finetune(model, num_epoch, train_ldr, val_ldr, best_acc, save_if=False):
    # initial varaibles both for record and visulization
    best_dict = model.state_dict()
    start_epoch = 0
    best_epoch = 0
    train_trace, val_trace = dict(), dict()
    train_trace['acc'], train_trace['loss'] = [], []
    val_trace['acc'], val_trace['loss'] = [], []
    writer = SummaryWriter(log_path)
    optimizer = optim.SGD(model.parameters(), lr=lr1, momentum=momentum, weight_decay=wd1)
    evaluator = torch.nn.CrossEntropyLoss()
    if scheduler_name == 'COSINE':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epoch)
    elif scheduler_name == 'STEP_LR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_interval, gamma=lr_scale)
    for epoch in tqdm(range(start_epoch, start_epoch + num_epoch)):
        train_acc, train_loss = runTrain(epoch, train_ldr, optimizer, model, evaluator, args=args)
        scheduler.step()
        val_acc, val_loss = runTest(val_ldr, model, evaluator, args=args)
        # saving checkpoint
        print('validation record:', val_loss, val_acc)
        if (val_acc > best_acc):
            with torch.no_grad():
                best_dict = copy.deepcopy(model.state_dict())
            best_acc = val_acc
            best_epoch = epoch
            print('Saving model..  with acc {0} in the epoch {1}'.format(best_acc, epoch))
            state = {
                'best_acc': best_acc,
                'best_epoch': epoch,
                'best_net': model.state_dict(),
                'best_net_with_class': model,
                'traces': {'train': train_trace, 'val': val_trace},
                'config': args
            }
            if save_if:
                torch.save(state, os.path.join(ckpt_path, model_name + '.pth'))
        # record and log
        train_trace['acc'].append(train_acc)
        train_trace['loss'].append(train_loss)
        val_trace['acc'].append(val_acc)
        val_trace['loss'].append(val_loss)
        # record in tensorboard
        writer.add_scalars('loss', {'val': val_loss, 'train': train_loss},
                           epoch + 1)
        writer.add_scalars('acc', {'val': val_acc, 'train': train_acc},
                           epoch + 1)
        print('Epoch %d: train acc %.5f, test acc %.5f ' % (epoch, train_acc, val_acc))
    print('Finish training: the best traning accuray is {} in epoch {}. \n The relate checkpoint path: {}'.format(
        best_acc,
        best_epoch,
        os.path.join(
            ckpt_path,
            model_name + '.pth')))
    return best_dict, best_acc


def extractThreshBFS(qcs_model, reset=False):
    # todo: whether there's a better method to guanrantee the order
    thresh_list = []
    name_list = []
    with torch.no_grad():
        queue = [qcs_model]
        idx = 0
        while len(queue) > 0:
            module = queue.pop()
            for name, child in module.named_children():
                if name == 'act_fun': continue  # skip the template
                if isinstance(child, (TrainableQCS, QCS, NoisyQCS, MovingQCS, TClamp, LeackyQCS, LeackyClamp)):
                    thresh_list.append(child.thresh.detach().item())
                    name_list.append(name)
                    if reset:
                        setattr(child, 'thresh', None)
                    # child.thresh = child.thresh.detach().clone()
                    idx += 1
                else:
                    queue.insert(0, child)
    return name_list, thresh_list


def extractThresh(root_model, thresh_list=[], name_list=[], reset=False, abs_if=False):
    # todo: whether there's a better method to guanrantee the order
    with torch.no_grad():
        for name, child in root_model.named_children():
            if name == 'act_fun': continue  # skip the template
            if isinstance(child,
                          (TrainableQCS, QCS, NoisyQCS, MovingQCS, NoisyQC, TClamp, NTClamp, LeackyQCS, LeackyClamp)):
                if abs_if:
                    thresh_list.append(abs(child.thresh.detach().item()))
                else:
                    thresh_list.append(child.thresh.detach().item())
                name_list.append(name)
                if reset:
                    setattr(child, 'thresh', None)
                # child.thresh = child.thresh.detach().clone()
            else:
                name_list, thresh_list = extractThresh(child, thresh_list, name_list, reset=reset, abs_if=abs_if)
    return name_list, thresh_list


def get_model_name(model_name):
    aug_str = '_'.join(['cut' if cutout else ''] + ['aug' if auto_aug else ''])
    if aug_str[0] != '_': aug_str = '_' + aug_str
    if aug_str[-1] != '_': aug_str = aug_str + '-'
    model_name += dataset.lower() + aug_str + 'ann' + '_' + architecture.lower() + '_act_' + act_fun_name + '-opt_' + optim_name.lower() + (
        '_bn' if use_bn else '') + ('_bias' if bias else '') + '_wd_' + str(
        wd1)
    if not enable_shift:
        model_name += '_noshift'
    cas_num = len([one for one in os.listdir(log_path) if one.startswith(model_name)])
    model_name += '_cas_' + str(cas_num)
    print('model name: ' + model_name)
    return model_name


def runPhaseI(ann, args, train_ldr, val_ldr, best_acc):
    global model_name
    best_dict, ann_acc = finetune(ann, args.num_epoch1, train_ldr, val_ldr, best_acc, save_if=(args.num_epoch1 > 0))
    ann.load_state_dict(best_dict)
    # search_fold_and_remove_bn(ann, replace=True)
    name_list, thresh_list = extractThresh(ann, thresh_list=[], reset=False, abs_if=True)
    thresh_list.append(1)
    print(thresh_list)
    snn = convertSNN(ann, thresh=thresh_list, record_interval=record_interval, args=args,
                     decay=torch.ones(num_perserve))
    # snn = convertSNN(ann, thresh=thresh_list, record_interval=True)
    # snn = IsomorphicSNN(copy.copy(ann), T, decay=torch.ones(num_perserve), thresh=torch.ones(num_perserve),
    #                     enable_shift=True,
    #                     mode='spike')
    # snn = SpikingVGG(architecture, num_class=num_class, dataset=args.dataset, dropout=0, feature_drop=0, bias=args.bias,
    #                  readout='mean_cum', neu_mode='spike', decay=torch.ones(num_perserve),
    #                  thresh=torch.tensor(thresh_list)).to(device, dtype)
    # transfer_dict = transfer_model(ann.state_dict(), snn.state_dict(), model=False)
    # snn.load_state_dict(transfer_dict, strict=False)
    snn_acc, _ = runTest(val_ldr, snn, nn.CrossEntropyLoss(), encoder=RateCoding(T, method=0), args=args)
    return (ann, ann_acc), (snn, snn_acc)


def getDataLdr():
    train_data, val_data, num_class = loadData(dataset, data_path, cutout=False,
                                               auto_aug=False)  # todo: try other data aug
    train_ldr = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.train_batch_size, shuffle=True,
                                            pin_memory=True, num_workers=num_workers)
    val_ldr = torch.utils.data.DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                                          pin_memory=True,
                                          num_workers=num_workers)
    if dataset == 'MNIST':
        return train_ldr, train_ldr, val_ldr, num_class
    train_data_, _, num_class = loadData(dataset, data_path, cutout=True,
                                         auto_aug=True)  #
    train_ldr_ = torch.utils.data.DataLoader(dataset=train_data_, batch_size=args.train_batch_size, shuffle=True,
                                             pin_memory=True, num_workers=num_workers)

    return train_ldr_, train_ldr, val_ldr, num_class


def loadModel(args):
    global num_perserve
    state = torch.load(args.pretrain_path, map_location=device)
    act_fun_args = {}
    act_fun = None
    if args.act_fun_name == 'relu':
        act_fun = nn.ReLU
    elif args.act_fun_name == 'clamp':
        act_fun = Clamp
    elif args.act_fun_name == 'qcs':
        act_fun = TrainableQCS
        act_fun_args['thresh'] = torch.tensor(1.0)
        act_fun_args['nb_step'] = args.T
    elif args.act_fun_name == 'noisy_qcs':
        act_fun = NoisyQCS
        act_fun_args = {'thresh': torch.tensor(1.0), 'nb_step': args.T, 'shift': 0, 'trainable': True,
                        'p': args.denoise_rate, 'detach': detach}
    elif args.act_fun_name == 'tclamp':
        act_fun = TClamp
    elif args.act_fun_name == 'nqc':
        act_fun = NoisyQC
        act_fun_args = {'thresh': torch.tensor(1.0), 'nb_steps': args.T,
                        'noise_prob': args.denoise_rate, 'abs_if': True}
    elif args.act_fun_name == 'lqcs':
        act_fun = LeackyQCS
        act_fun_args = {
            'thresh': torch.tensor(5.0),
            'nb_steps': args.T,
        }
    elif args.act_fun_name == 'lclamp':
        act_fun = LeackyClamp
    elif args.act_fun_name == 'lrelu':
        act_fun = nn.LeakyReLU
    if args.act_fun_name == 'ntclamp':
        act_fun = NTClamp

    if 'VGG' in args.architecture:
        ann = rVGG(architecture, num_class, args.dataset, act_fun=act_fun(**act_fun_args), bias=args.bias,
                   feature_drop=args.feature_drop,
                   dropout=args.classifier_drop,
                   use_bn=args.use_bn, hook=False).to(device, dtype)
    elif 'Res' in args.architecture:
        depth = int(re.findall("\d+", args.architecture)[0])
        num_perserve = 3 * depth + 3 if depth >= 50 else 2 * depth + 3
        ann = ResNetX(depth, num_class=num_class, act_fun=act_fun(**act_fun_args),
                      modified='modified' in args.architecture).to(device, dtype)
        if not args.use_bn:
            search_fold_and_remove_bn(ann)
    elif 'LeNet' in args.architecture:
        ann = LeNet5(act_fun=act_fun(**act_fun_args)).to(device, dtype)
        if not args.use_bn:
            search_fold_and_remove_bn(ann)
    elif 'mobilev2' == args.architecture:
        num_perserve = 53
        ann = MobileNetV2(width_mult=1, n_class=10, input_size=32, act_fun=act_fun(**act_fun_args)).to(device, dtype)
        if not args.use_bn:
            search_fold_and_remove_bn(ann)
    elif 'resnetxt' == args.architecture:
        num_perserve = 30
        ann = ResNeXt29_2x64d(act_fun=act_fun(**act_fun_args)).to(device, dtype)

    # load model and test results
    # ann.act_fun = None
    try:
        ann.load_state_dict(state['best_net'])
    except RuntimeError:
        print('The saved structure and running net structure are mismatching, try sequential weights matching ... ')
        transfer_dict = transfer_model(state['best_net'], ann.state_dict(), model=False)
        ann.load_state_dict(transfer_dict, strict=False)
    ann.act_fun = act_fun(**act_fun_args)
    return ann, act_fun


def convertSNN(ann, args, decay, thresh, record_interval=None):
    # todo: add a BN checker for alpah=1 & beta=0 or fuse bn & replace it
    ann_ = copy.deepcopy(ann)
    replaceAct(ann_, nn.BatchNorm2d, StraightThrough(), rename_act=False)
    snn = IsomorphicSNN(ann_, args.T, decay=decay, thresh=thresh,
                        enable_shift=enable_shift, spike_fn=RectSurrogate.apply,
                        mode='psp', enable_init_volt=not enable_shift, record_interval=record_interval,
                        specials=specials, cell=AIFCell)
    return snn


#
# def main():
#     # load args and setting some constant hyperparameters
#     # todo: test phase I, Phase II, Phase I & Phase II individually
#     from models.resnet import resnet_specials
#     specials = resnet_specials
#     global device, dtype
#     setup_seed(seed)
#     num_workers = 0
#     num_perserve = 16
#     dtype = torch.float
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     print(device)
#     start_epoch = 0
#     args.device, args.dtype, args.start_epoch = device, dtype, start_epoch
#     print('Start running on device {}'.format(device))
#     # Initialization
#     train_ldr_, train_ldr, val_ldr, num_class = getDataLdr()
#     ann, act_fun = loadModel(args)
#     print('Pretrained ANN with {}:'.format(act_fun_name), runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args))
#     search_fold_and_remove_bn(ann, replace=True)
#     print('ANN after Folding BN into Weights:', runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args))
#     if args.act_fun_name == 'relu':
#         args.percentile = 0.995
#         name, max_acts = getMaxAct(ann, train_ldr, hook_class=nn.ReLU, device=device, dtype=dtype,
#                                    percentile=args.percentile)
#         balanceWeight(ann, name, max_acts)
#         print('ANN after Weight Balancing with percentile factor {}:'.format(args.percentile),
#               runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args))
#         replaceAct(ann, act_fun,
#                    [NoisyQCS(torch.tensor(1.0), T, p=0.5, shift=0, trainable=True) for i in range(num_perserve)])
#     elif args.act_fun_name == 'noisy_qcs':
#         pass
#     else:
#         raise NotImplementedError("not organized into this version")
#     # replaceAct(ann, act_fun,
#     #            [QCS(torch.tensor(1.0), T,  shift=0) for i in range(num_perserve)])
#     best_acc, _ = runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args)
#     print('ANN after Quantitation and Clamping : ', best_acc)
#     snn = convertSNN(ann, args, decay=torch.ones(num_perserve), thresh=torch.ones(num_perserve), record_interval=2)
#     print('SNN with Shared Weights and Bias:',
#           runTest(val_ldr, snn, nn.CrossEntropyLoss(), encoder=RateCoding(T, method=0), args=args))
#
#     # Start Running Phase I
#     cutout = auto_aug = True
#     momentum = 0.9
#     lr_scale = 0.1
#     lr_interval = [int(ratio * num_epoch1) for ratio in args.lr_interval]
#     log_path = './log'
#     scheduler_name = 'STEP_LR'
#     args.log_interval = 100
#     args.num_epoch = args.num_epoch1
#     runPhaseI(ann, args, train_ldr_, val_ldr, best_acc)
#     # Start Running Phase II
#
#     # todo: merge those phases into one script
#

def sampleData(data_ldr, num_samples):
    input_samples = []
    target_samples = []
    for (input, target) in data_ldr:
        input_samples.append(input)
        target_samples.append(target)
        if len(input_samples) * len(input_samples[0]) >= num_samples:
            break
    input_samples = torch.cat(input_samples, dim=0)[:num_samples]
    target_samples = torch.cat(target_samples, dim=0)[:num_samples]
    return input_samples, target_samples


# @profile
def calibInitMemPoten(ann, snn, samples, batch_size=-1, ann_hook_cls=nn.ReLU, snn_hook_cls=LIFLayer,
                      encoder=RateCoding(method=0, nb_steps=16)):
    ann.eval()
    snn.eval()
    device = samples.device
    with torch.no_grad():
        snn_cali_layers, ann_cali_layers = [], []
        for n, m in ann.named_modules():
            if n == 'act_fun': continue
            if isinstance(m, ann_hook_cls):
                ann_cali_layers.append(m)
        for m in snn.modules():
            if isinstance(m, snn_hook_cls):
                snn_cali_layers.append(m)
        assert len(snn_cali_layers) == len(
            ann_cali_layers), 'The number of source layers and target layers is mismatching ...'
        for i in range(len(ann_cali_layers)):
            ann_layer = ann_cali_layers[i]
            snn_layer = snn_cali_layers[i]
            # print(i, ann_layer.thresh, snn_layer.thresh)
            assert abs(ann_layer.thresh) == abs(
                snn_layer.thresh), "The threshes in the ann layer and snn layer are mismatching"
            ann_hooker = getActs(ann, ann_layer, samples, batch_size=args.acts_bz, to_cpu=True)
            # spike_samples = encoder(samples)
            snn_hooker = getActs(snn, snn_layer, samples, encoder=encoder, batch_size=args.acts_bz, to_cpu=True)
            # snn_hooker2 = getActs(snn, snn_layer, samples, encoder=encoder, batch_size=-1)
            # print((snn_hooker.outputs != snn_hooker2.outputs).sum())
            # print(snn_layer)
            # print(torch.where(snn_hooker.outputs != snn_hooker2.outputs))
            torch.cuda.empty_cache()
            ann_out = ann_hooker.outputs.to(device)
            snn_out = snn_hooker.outputs.mean(axis=0).to(device)
            snn_layer.init_mem += (ann_out.mean(axis=0) - snn_out.mean(
                axis=0)) * encoder.nb_steps
            del ann_hooker
            del snn_hooker


def state_forward(snn_module, x, ext_x=0):
    snn_module.reset_membrane_potential()
    out = []
    scale = 1
    if hasattr(snn_module, 'scale'):
        scale = snn_module.scale
    for t in range(len(x)):
        out.append(snn_module(x[t], scale * ext_x[t]))
    out = torch.stack(out)
    return out.mean(axis=0)


def getActs(model, layer, samples, batch_size=-1, encoder=None, to_cpu=False):
    # todo: strange bug: the results are different when  batch_size = -1 and batch_size = X on gpu, howerver same on cpu
    model.eval()
    layer.eval()
    if batch_size == -1: batch_size = len(samples)
    with torch.no_grad():
        hooker = RecordHook(to_cpu=to_cpu)
        handler = layer.register_forward_hook(hooker)
        # if batch_size == -1:
        #     if encoder is not None:
        #         samples = encoder(samples)
        #     model(samples)
        #     if encoder is None:
        #         hooker.inputs = hooker.inputs[0]
        #         hooker.outputs = hooker.outputs[0]
        #         hooker.extern_inputs = hooker.extern_inputs[0]
        #     else:
        #         hooker.inputs = torch.stack(hooker.inputs)
        #         hooker.outputs = torch.stack(hooker.outputs)
        #         hooker.extern_inputs = torch.stack(hooker.extern_inputs)
        # else:
        num = samples.shape[0]
        index = torch.arange(0, num)
        inputs, outputs, externs = [], [], []
        for i in range(int(math.ceil(num / batch_size))):
            batch_index = index[i * batch_size:min((i + 1) * batch_size, num)]
            batch_samples = samples[batch_index]
            if encoder is not None:
                batch_samples = encoder(batch_samples)
                model(batch_samples)
                batch_inputs, batch_outputs, batch_externs = hooker.reset()
                inputs.append(batch_inputs)
                outputs.append(batch_outputs)
                externs.append(batch_externs)
            else:
                model(batch_samples)
        if encoder is None:
            hooker.inputs = torch.cat(hooker.inputs)
            hooker.outputs = torch.cat(hooker.outputs)
            hooker.extern_inputs = torch.cat(hooker.extern_inputs)
        else:
            hooker.inputs = torch.cat(inputs, axis=1)
            hooker.outputs = torch.cat(outputs, axis=1)
            hooker.extern_inputs = torch.cat(externs, axis=1)
        handler.remove()
    return hooker


def cosine_loss(batch_output, batch_target):
    return 1 - torch.cosine_similarity(batch_output.view(-1, 1, 1, 1).squeeze(),
                                       batch_target.view(-1, 1, 1, 1).squeeze(), dim=0)


def kl_loss(batch_output, batch_target):
    batch_size = batch_output.shape[0]
    batch_output = batch_output.view(batch_size, -1)
    batch_target = batch_target.view(batch_size, -1)
    div_loss = nn.KLDivLoss()
    prob_output = F.log_softmax(batch_output, dim=-1)
    prob_target = F.softmax(batch_target, dim=-1)
    return div_loss(prob_output, prob_target)


def js_loss(net_1_logits, net_2_logits):
    # todo: check error
    raise NotImplementedError('Still have some strange error')
    net_1_probs = F.softmax(net_1_logits, dim=-1)
    net_2_probs = F.softmax(net_2_logits, dim=-1)
    m = 0.5 * (net_1_probs + net_1_probs)
    loss = 0.0
    loss += F.kl_div(m.log(), net_1_logits, reduction="mean")
    loss += F.kl_div(m.log(), net_2_logits, reduction="mean")
    return 0.5 * loss


def caliWeightBPTT(ann, snn, train_samples, val_samples, batch_size=1, ann_hook_cls=nn.ReLU, snn_hook_cls=LIFLayer,
                   encoder=RateCoding(method=0, nb_steps=16), args=None):
    snn_cali_layers, ann_cali_layers = [], []
    for n, m in ann.named_modules():
        if n == 'act_fun': continue
        if isinstance(m, ann_hook_cls):
            ann_cali_layers.append(m)
    for m in snn.modules():
        if isinstance(m, snn_hook_cls):
            snn_cali_layers.append(m)
    assert len(snn_cali_layers) == len(
        ann_cali_layers), 'The number of source layers and target layers is mismatching ...'
    num_train = len(train_samples)
    num_val = len(val_samples)
    for i in range(len(ann_cali_layers)):
        snn.eval()
        ann.eval()
        ann_layer, snn_layer = ann_cali_layers[i], snn_cali_layers[i]
        ann_train_hooker = getActs(ann, ann_layer, train_samples, batch_size=args.acts_bz, to_cpu=True)
        snn_train_hooker = getActs(snn, snn_layer, train_samples, encoder=encoder, batch_size=acts_bz, to_cpu=True)
        train_inputs = snn_train_hooker.inputs
        train_extins = snn_train_hooker.extern_inputs
        train_targets = ann_train_hooker.outputs  # todo: check here
        torch.cuda.empty_cache()

        ann_val_hooker = getActs(ann, ann_layer, val_samples, batch_size=args.acts_bz, to_cpu=True)
        snn_val_hooker = getActs(snn, snn_layer, val_samples, encoder=encoder, batch_size=args.acts_bz, to_cpu=True)
        val_inputs = snn_val_hooker.inputs
        val_extins = snn_val_hooker.extern_inputs
        val_targets = ann_val_hooker.outputs  # todo: check here
        torch.cuda.empty_cache()
        snn_layer.init_mem = nn.Parameter(snn_layer.init_mem, requires_grad=True)
        snn_layer.spike_fn = NoisySpike(sig='tanh')
        snn_layer = snn_layer.to(args.device_name)

        print([n for n, v in snn_layer.named_parameters()])
        if args.optim_name2 == 'ADAM':
            optimizer = optim.Adam(snn_layer.parameters(), lr=args.lr2, weight_decay=args.wd2)
            # optimizer = optim.Adam([{'params': snn_layer.func.parameters()}], lr=args.lr2, weight_decay=args.wd2,
            #                        amsgrad=True)
        elif args.optim_name2 == 'SGDM':
            optimizer = optim.SGD(snn_layer.parameters(), lr=args.lr2, weight_decay=args.wd2,
                                  momentum=0.9)
            # optimizer = optim.SGD([{'params': snn_layer.func.parameters()}], lr=args.lr2, weight_decay=args.wd2,
            #                       momentum=0.9)
        elif args.optim_name2 == 'RADAM':
            optimizer = RAdam(snn_layer.parameters(), lr=args.lr2, weight_decay=args.wd2)

            # optimizer = RAdam([{'params': snn_layer.func.parameters()}], lr=args.lr2, weight_decay=args.wd2)
        # print(args.lr2)
        # optimizer = optim.SGD([snn_layer.func.weight], lr=snn_lr, weight_decay=snn_wd, momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch2, eta_min=0.)

        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        if warmup_step > 0:
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_step,
                                                      after_scheduler=scheduler)

        # evaluator = nn.MSELoss()
        # evaluator = nn.SmoothL1Loss()
        # evaluator = cosine_loss
        evaluator = kl_loss
        # evaluator = js_loss

        # evaluator = torch.cosine_similarity
        stoper = EarlyStoping(patience=patience, verbose=False)
        print('Start ajusting weight with bptt for {}-th layer'.format(i))
        for iepoch in range(args.num_epoch2):
            if warmup_step > 0:
                scheduler_warmup.step(iepoch)
            index = torch.randperm(num_train)
            train_loss_tot = 0
            snn.train()
            snn_layer.train()
            for ibatch in range(math.ceil(num_train / batch_size)):
                # if (ibatch + 1) * batch_size > num_samples: break
                batch_index = index[ibatch * batch_size:min((ibatch + 1) * batch_size, num_train)]
                batch_input = train_inputs[:, batch_index]
                batch_extin = train_extins[:, batch_index]
                batch_target = train_targets[batch_index]
                optimizer.zero_grad()
                batch_input = batch_input.to(device, dtype)
                batch_target = batch_target.to(device, dtype)
                batch_extin = batch_extin.to(device, dtype)
                # batch_output = snn_layer(batch_input).mean(axis=0)
                batch_output = state_forward(snn_layer, batch_input, batch_extin)
                loss = evaluator(batch_output, batch_target)
                # loss = 1 - evaluator(batch_output.view(-1, 1, 1, 1).squeeze(),
                #                      batch_target.view(-1, 1, 1, 1).squeeze(), dim=0)
                train_loss_tot += loss.detach().cpu()
                loss.backward()
                optimizer.step()
            if warmup_step == 0:
                scheduler.step(train_loss_tot)
            snn.eval()
            snn_layer.eval()
            val_loss_tot = 0
            with torch.no_grad():
                index = torch.arange(0, num_val)
                for ibatch in range(math.ceil(num_val / batch_size)):
                    # if (ibatch + 1) * batch_size > num_samples: break
                    batch_index = index[ibatch * batch_size:min((ibatch + 1) * batch_size, num_val)]
                    batch_input = val_inputs[:, batch_index]
                    batch_extin = val_extins[:, batch_index]
                    batch_target = val_targets[batch_index]
                    batch_input = batch_input.to(device, dtype)
                    batch_target = batch_target.to(device, dtype)
                    batch_extin = batch_extin.to(device, dtype)
                    batch_output = state_forward(snn_layer, batch_input, batch_extin)
                    loss = evaluator(batch_output, batch_target)
                    # loss = 1 - evaluator(batch_output.view(-1, 1, 1, 1).squeeze(),
                    #                      batch_target.view(-1, 1, 1, 1).squeeze(), dim=0)
                    val_loss_tot += loss.cpu()
            if iepoch % 20 == 0:
                print('Epoch {} with train loss {}, validation loss {}'.format(iepoch, train_loss_tot, val_loss_tot))
            stop_if = stoper(val_loss_tot, snn_layer)
            if (stop_if):
                print('Early Stopping at Epoch {}'.format(iepoch))
                break
        snn_layer.load_state_dict(stoper.best_state_dict)
        #  todo: clear those hooker by function namespace
        del ann_val_hooker
        del ann_train_hooker
        del snn_train_hooker
        del snn_val_hooker


def runPhaseII(ann, snn, train_ldr, val_ldr, args):
    best_net = None
    samples, _ = sampleData(train_ldr, args.num_samples)
    samples = samples.to(args.device, args.dtype)
    val_samples, _ = sampleData(train_ldr, args.num_samples // 4)  # num of train samples vs. num of val samples = 4 : 1
    val_samples = val_samples.to(args.device, args.dtype)

    calibInitMemPoten(ann, snn, samples, ann_hook_cls=type(ann.act_fun), snn_hook_cls=AIFCell,
                      encoder=RateCoding(method=0, nb_steps=T))
    snn_acc1, _ = runTest(val_ldr, snn, nn.CrossEntropyLoss(), args=args, encoder=RateCoding(method=0, nb_steps=T))
    best_acc, best_net = snn_acc1[T], copy.deepcopy(snn)

    print('SNN with Average Membrane Potential Calibration:',
          snn_acc1[T])
    caliWeightBPTT(ann, snn, samples, val_samples, batch_size=args.bz2, ann_hook_cls=type(ann.act_fun),
                   snn_hook_cls=AIFCell,
                   encoder=RateCoding(method=0, nb_steps=T), args=args)
    snn_acc2, _ = runTest(val_ldr, snn, nn.CrossEntropyLoss(), args=args, encoder=RateCoding(method=0, nb_steps=T))
    print('SNN with BPTT Calibration:',
          snn_acc2[T])
    if snn_acc2[T] > best_acc:
        best_acc, best_net = snn_acc2[T], copy.deepcopy(snn)
    calibInitMemPoten(ann, snn, samples, ann_hook_cls=type(ann.act_fun), snn_hook_cls=AIFCell,
                      encoder=RateCoding(method=0, nb_steps=T))
    snn_acc3, _ = runTest(val_ldr, snn, nn.CrossEntropyLoss(), args=args, encoder=RateCoding(method=0, nb_steps=T))
    print('SNN with second Average Membrane Potential Calibration:',
          snn_acc3[T])
    if snn_acc3[T] > best_acc:
        best_acc, best_net = snn_acc3[T], copy.deepcopy(snn)
    return (snn_acc1[T], snn_acc2[T], snn_acc3[T], best_acc), best_net


if __name__ == '__main__':
    # todo: add abs_if into hyperparameters
    from vgg_convert_config import *
    from models.resnet import resnet_specials
    from models.resnetxt import resnetxt_specials
    from models.mobilenet import mobilenet_specials

    specials = {
        'VGG16': {},
        'Res18': resnet_specials,
        'mobilev2': mobilenet_specials,
        'resnetxt': resnetxt_specials
    }
    print('Hyper Parameter Config:')
    for k, v in vars(args).items():
        print(' --' + str(k) + ' ' + str(v))
    args.cmd = ['python ' + os.path.basename(__file__)] + args.cmd
    results['cmd'] = ' '.join(args.cmd)
    specials = specials[args.architecture]
    globals().update(vars(args))
    global device, dtype
    setup_seed(seed)
    RectSurrogate.alpha = 1  # todo: adjust the width
    num_workers = 0
    num_perserve = 16
    dtype = torch.float

    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda:1')
    start_epoch = 0
    cutout = auto_aug = True
    momentum = 0.9
    lr_scale = 0.1
    args.device, args.dtype, args.start_epoch = device, dtype, start_epoch
    model_name = get_model_name('finetune_T_' + str(args.T) + '-')
    results['model_name'] = model_name
    if not os.path.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    log_path = os.path.join(args.log_path, model_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    sys.stdout = Logger(os.path.join(log_path, model_name + '.log'))
    sys.stderr = Logger(os.path.join(log_path, model_name + '.log'))

    print('Start running on device {}'.format(device))
    # Initialization
    train_ldr_, train_ldr, val_ldr, num_class = getDataLdr()
    ann, act_fun = loadModel(args)
    results['ann'], _ = runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args)
    print('Pretrained ANN with {}:'.format(act_fun_name), results['ann'])
    # todo: add supports for replace = False [finish]
    search_fold_and_remove_bn(ann, replace=True)
    # search_fold_and_reset_bn(ann) # update
    # snn_no_bn = copy.deepcopy(ann)
    # search_fold_and_remove_bn(snn_no_bn, replace=True)
    results['fold_ann'], _ = runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args)
    print('ANN after Folding BN into Weights:', results['fold_ann'])
    print(ann)
    if args.act_fun_name == 'relu' or args.act_fun_name == 'clamp' or args.act_fun_name == 'lrelu':
        if args.act_fun_name == 'relu' or args.act_fun_name == 'clamp':
            args.percentile = 0.995  # empirical config
            name, max_acts = getMaxAct(ann, train_ldr, hook_class=nn.ReLU, device=device, dtype=dtype,
                                       percentile=args.percentile)
            balanceWeight(ann, name, max_acts)
            print('ANN after Weight Balancing with percentile factor {}:'.format(args.percentile),
                  runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args))
            replaceAct(ann, act_fun,
                       [NoisyQCS(torch.tensor(1.0), T, p=denoise_rate, shift=0, trainable=True, detach=detach) for i in
                        range(num_perserve)])
        if args.act_fun_name == 'lrelu':
            args.percentile = 0.995  # empirical config
            name, max_acts = getMaxAct(ann, train_ldr, hook_class=nn.LeakyReLU, device=device, dtype=dtype,
                                       percentile=args.percentile)
            balanceWeight(ann, name, max_acts)
            print('ANN after Weight Balancing with percentile factor {}:'.format(args.percentile),
                  runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args))

            replaceAct(ann, act_fun,
                       [LeackyQCS(torch.abs(torch.tensor(1.0)), args.T) for i in
                        range(num_perserve)])

    elif args.act_fun_name == 'tclamp':
        name_list, thresh_list = extractThreshBFS(ann)
        print(thresh_list)
        # print(name_list)
        replaceAct(ann, TClamp,
                   [NoisyQC(torch.abs(torch.tensor(thresh)), args.T, noise_prob=denoise_rate,
                            detach=detach) for thresh in
                    thresh_list])
    elif args.act_fun_name == 'noisy_qcs' or args.act_fun_name == 'nqc' or args.act_fun_name == 'lqcs':
        pass
    elif args.act_fun_name == 'lclamp':
        name_list, thresh_list = extractThreshBFS(ann)
        print(thresh_list)
        # print(name_list)
        replaceAct(ann, LeackyClamp,
                   [LeackyQCS(torch.abs(torch.tensor(thresh)), args.T) for thresh in
                    thresh_list])
    else:
        raise NotImplementedError("not organized into this version")
    # replaceAct(ann, act_fun,
    #            [QCS(torch.tensor(1.0), T,  shift=0) for i in range(num_perserve)])

    best_acc, _ = runTest(val_ldr, ann, nn.CrossEntropyLoss(), args=args)
    results['qc_ann'] = best_acc
    print('ANN after Quantitation and Clamping : ', best_acc)
    name_list, thresh_list = extractThresh(ann, thresh_list=[], abs_if=True)
    thresh_list.append(1)
    # thresh_list = [abs(thresh) for thresh in thresh_list]
    print(len(thresh_list), thresh_list)
    # print(name_list)
    snn = convertSNN(ann, args, decay=torch.ones(num_perserve), thresh=thresh_list,
                     record_interval=record_interval)

    snn_acc, _ = runTest(val_ldr, snn, nn.CrossEntropyLoss(), encoder=RateCoding(T, method=0), args=args)
    print('SNN with Shared Weights and Bias:',
          snn_acc)
    results['ann2snn'] = snn_acc[T]

    # Start Running Phase I

    lr_interval = [int(ratio * num_epoch1) for ratio in args.lr_interval]
    # log_path = './log'
    scheduler_name = 'COSINE'  # todo: adjust here
    args.log_interval = 100
    args.num_epoch = args.num_epoch1
    (ann, results['fintune_qc_ann']), (snn, results['fintune_qc_ann2snn']) = runPhaseI(ann, args, train_ldr_, val_ldr,
                                                                                       best_acc)
    results['fintune_qc_ann2snn'] = results['fintune_qc_ann2snn'][T]
    print('SNN with Shared Weights and Bias:', results['fintune_qc_ann2snn'])
    # Start Running Phase II
    if args.num_epoch2 == 0:
        print('Phase II is skipped as num_epoch2 ')
        exit(1)
    (results['pc_snn'], results['wc_pc_snn'], results['pc_wc_pc_snn'], best_acc_rec), best_net = runPhaseII(
        copy.deepcopy(ann), copy.deepcopy(snn), train_ldr, val_ldr, args)
    # lr2_rec = args.lr2
    # print('Result: {} {}'.format(args.lr2, best_acc_rec))
    state = {
        'best_acc': best_acc_rec,
        'best_state': best_net.state_dict(),
        'best_net': best_net,
        'config': args,
        'abstract': results,
        'thresh': thresh_list
    }
    model_name = model_name.replace('ann', 'snn')
    print('Saving best net in path: ', model_name)
    torch.save(state, os.path.join(ckpt_path, model_name + '.pth'))
    if args.grid_search is not None:
        write_head = not os.path.exists(args.grid_search)
        with open(args.grid_search, 'a+') as f:

            writer = csv.DictWriter(f, fieldnames=list(results.keys()))
            if write_head:
                writer.writeheader()
            # writer.writeheader()
            writer.writerow(results)
