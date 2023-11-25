"""
-*- coding: utf-8 -*-

@Time    : 2021/4/26 15:16

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : surrogate_act.py

Funcition: In this file, I want to provide some surrogate gradient function for both direct training for snn and ann2snn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateHeaviside(torch.autograd.Function):
    # Activation function with surrogate gradient
    sigma = 10.0

    @staticmethod
    def forward(ctx, input):
        output = torch.zeros_like(input)
        output[input > 0] = 1.0
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # approximation of the gradient using sigmoid function
        grad = grad_input * torch.sigmoid(SurrogateHeaviside.sigma * input) * torch.sigmoid(
            -SurrogateHeaviside.sigma * input)
        return grad


class RectSurrogate(torch.autograd.Function):
    """
    activation function: rectangular function h(*)
    """
    alpha = 0.8

    @staticmethod
    def forward(ctx, input):
        """
           input = vin -thresh
        """
        # output = torch.zeros_like(input)
        # output[input > 0] = 1.0
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, delta):
        vin, = ctx.saved_tensors
        # delta_tmp = delta.clone() ####???????
        dgdv = 1.0 / RectSurrogate.alpha * (torch.abs(vin) < (RectSurrogate.alpha / 2.0))
        return delta * dgdv


class PDFSurrogate(torch.autograd.Function):
    alpha = 0.1
    beta = 0.1

    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = PDFSurrogate.alpha * torch.exp(-PDFSurrogate.beta * torch.abs(inpt))
        return sur_grad * grad_input


class LinearSurrogate(torch.autograd.Function):
    '''
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    '''
    gamma = 0.3  # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = LinearSurrogate.gamma * F.threshold(1.0 - torch.abs(inpt), 0, 0)
        return grad_input * sur_grad.float()


class Clamp(nn.Module):
    def __init__(self, min=0.0, max=1.0):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)


class TClamp(nn.Module):
    def __init__(self, min=0.0, max=1.0, thresh=5.0):
        super(TClamp, self).__init__()
        self.min = min
        self.max = max
        self.thresh = nn.Parameter(torch.tensor(thresh))

    def forward(self, x):
        # self.clamp()
        thresh = torch.abs(self.thresh)
        return thresh * torch.clamp(x / thresh, min=self.min, max=self.max)

    # def clamp(self):
    #     self.thresh.clamp_(min=0.)


class NTClamp(nn.Module):
    def __init__(self, min=0.0, max=1.0, thresh=1.0):
        super(NTClamp, self).__init__()
        self.min = min
        self.max = max
        self.thresh = nn.Parameter(torch.tensor(thresh))

    def forward(self, x):
        # self.clamp()
        # thresh = torch.abs(self.thresh)
        return self.thresh * torch.clamp(x / self.thresh, min=self.min, max=self.max)


class Shift(nn.Module):
    def __init__(self, shift_len):
        super(Shift, self).__init__()
        self.shift_len = shift_len

    def forward(self, input):
        return input + self.shift_len


class RoundSurrogate(torch.autograd.Function):
    '''
     Replace the partial of Round Function with a dydx= 1
    '''

    @staticmethod
    def forward(ctx, input):
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class RoundSurrogateII(torch.autograd.Function):
    '''
        Replace the partial of Round Function with a piecewise function
    '''
    scale = 4.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return torch.abs(input - input.round().float()) * RoundSurrogateII.scale * grad_output


class Quantize(nn.Module):
    # round and ceil could converted into each other by shift
    def __init__(self, factor, method=2):
        super(Quantize, self).__init__()
        self.factor = factor
        self.method = method

    def forward(self, x):
        if (self.method == 1):
            x = torch.div(torch.round(torch.mul(x, self.factor)),
                          self.factor)
        elif (self.method == 2):
            x = torch.div(RoundSurrogate.apply(torch.mul(x, self.factor)),
                          self.factor)
        elif (self.method == 3):
            x = torch.div(RoundSurrogateII.apply(torch.mul(x, self.factor)),
                          self.factor)

        return x


class NoisyQuantize(nn.Module):
    def __init__(self, p, detach=False, **kwargs):
        super(NoisyQuantize, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("noisy probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.quantize = Quantize(**kwargs)
        self.detach = detach

    def create_mask(self, x: torch.Tensor):
        return F.dropout(torch.ones_like(x.data), self.p, training=True)

    def forward(self, x):
        # torch.cuda.empty_cache()  # empty the cache of mask
        self.quantize.training = self.training
        if self.training:
            mask = self.create_mask(x)
            noise = (self.quantize(x) - x) * mask
            if self.detach:
                return x + noise.detach()
            else:
                return x + noise
        return self.quantize(x)


# todo: Test the effectness of NosiyQuantizaeII
class NoisyQuantizeII(nn.Module):
    def __init__(self, p, detach=False, **kwargs):
        super(NoisyQuantizeII, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("noisy probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        # self.quantize = Quantize(**kwargs)
        self.detach = detach
        self.factor = kwargs['factor']

    def create_mask(self, x: torch.Tensor):
        return torch.bernoulli(torch.ones_like(x) * (1 - self.p))

    def forward(self, x):
        torch.cuda.empty_cache()  # empty the cache of mask
        # self.quantize.training = self.training
        if self.training:
            mask = self.create_mask(x)

            return RoundSurrogate.apply(x * self.factor) / self.factor * mask + (torch.logical_not(mask)) * x
            # torch.cuda.empty_cache()  # empty the cache of temp
            # return x
            # return self.quantize(x)[mask]+x[torch.logical_not(mask)]
            # noise = (self.quantize(x) - x) * mask
            if self.detach:
                return x + noise.detach()
            else:
                return x + noise
        return RoundSurrogate.apply(x * self.factor) / self.factor


class NoisyQCSII(nn.Module):
    '''
       Noisy Quantize Clip & Shift Module as auxiliary layer in ANN2SNN
    '''

    def __init__(self, thresh, nb_step, shift=None, trainable=True, p=0.5, detach=False):
        super(NoisyQCSII, self).__init__()
        if trainable:
            self.thresh = nn.Parameter(torch.tensor(thresh))
        else:
            self.thresh = thresh
        self.nb_step = nb_step
        self.shift = shift
        self.p = p
        self.quantize = None
        self.detach = detach
        # if shift is None:
        #     self.shift = Shift(-self.thresh / (2 * nb_step))  # default floor function
        # else:
        #     self.shift = Shift(shift)
        # # self.shift = Shift(0)
        # # self.scale_factor = self.thresh
        # self.clip = Clamp()

    def forward(self, x):
        quantize = NoisyQuantizeII(self.p, factor=self.nb_step / self.thresh,
                                   method=2, detach=self.detach)  # move all the non-leaf node into forward function
        quantize.training = self.training  # update the train/test status explicitly as the module doesn't register as a member
        if self.shift is None:
            shift = Shift(-self.thresh / (2 * self.nb_step))
        else:
            shift = Shift(self.shift)
        clip = Clamp()
        return clip(quantize(shift(x)) / self.thresh)


class NoisyQCS(nn.Module):
    '''
       Noisy Quantize Clip & Shift Module as auxiliary layer in ANN2SNN
    '''

    def __init__(self, thresh, nb_step, shift=None, trainable=True, p=0.5, detach=False):
        super(NoisyQCS, self).__init__()
        if trainable:
            self.thresh = nn.Parameter(torch.tensor(thresh))
        else:
            self.thresh = thresh
        self.nb_step = nb_step
        self.shift = shift
        self.p = p
        self.quantize = None
        self.detach = detach
        # if shift is None:
        #     self.shift = Shift(-self.thresh / (2 * nb_step))  # default floor function
        # else:
        #     self.shift = Shift(shift)
        # # self.shift = Shift(0)
        # # self.scale_factor = self.thresh
        # self.clip = Clamp()

    def forward(self, x):
        quantize = NoisyQuantize(self.p, factor=self.nb_step / self.thresh,
                                 method=2, detach=self.detach).to(
            device=x.device)  # move all the non-leaf node into forward function
        quantize.training = self.training  # update the train/test status explicitly as the module doesn't register as a member
        if self.shift is None:
            shift = Shift(-self.thresh / (2 * self.nb_step))
        else:
            shift = Shift(self.shift)
        clip = Clamp()
        return clip(quantize(shift(x)) / self.thresh)


class QCS(nn.Module):
    '''
       Quantize Clip & Shift Module as auxiliary layer in ANN2SNN
    '''

    def __init__(self, thresh, nb_step, shift=None, trainable=False, ):
        super(QCS, self).__init__()
        if trainable:
            self.thresh = nn.Parameter(torch.tensor(thresh))
        else:
            self.thresh = thresh
        self.nb_step = nb_step
        self.quantize = Quantize(nb_step / self.thresh)
        if shift is None:
            self.shift = Shift(-self.thresh / (2 * nb_step))
        else:
            self.shift = Shift(shift)
        # self.shift = Shift(0)
        # self.scale_factor = self.thresh
        self.clip = Clamp()

    def forward(self, x):
        return self.clip(self.quantize(self.shift(x)) / self.thresh)


class LeackyQCS(nn.Module):
    def __init__(self, thresh, nb_steps):
        super(LeackyQCS, self).__init__()
        self.thresh = nn.Parameter(torch.tensor(thresh))
        self.nb_step = nb_steps
        self.leacky_relu = nn.LeakyReLU()

    def forward(self, x):
        thresh = torch.abs(self.thresh)
        return torch.clip(RoundSurrogate.apply(self.leacky_relu(x) * self.nb_step / thresh) / self.nb_step, -1,
                          1) * thresh


class LeackyClamp(nn.Module):
    def __init__(self, thresh=5.0):
        super(LeackyClamp, self).__init__()
        self.leacky_relu = nn.LeakyReLU()
        self.thresh = nn.Parameter(torch.tensor(thresh))
        self.clip = Clamp(min=-1, max=1)

    def forward(self, x):
        thresh = torch.abs(self.thresh)
        return self.clip(self.leacky_relu(x) / thresh) * thresh


class TrainableQCS(nn.Module):
    '''
       Quantize Clip & ReScale Module as auxiliary layer in ANN2SNN
    '''

    def __init__(self, thresh, nb_step):
        super(TrainableQCS, self).__init__()
        self.thresh = nn.Parameter(thresh)
        self.nb_step = nb_step
        # self.clip = Clamp()

    def forward(self, x):
        return torch.clip(RoundSurrogate.apply(x * self.nb_step / self.thresh) / self.nb_step, 0, 1)
        # quantize = Quantize(self.nb_step / self.thresh)
        # quantize.training = self.training
        # self.shift = Shift(-self.thresh / (2 * self.nb_step))
        # self.shift = Shift(0)
        # self.scale_factor = self.thresh


class MovingQCS(nn.Module):
    def __init__(self, nb_step, percentile=None, momentum=None, scale_mode='layer_wise'):
        super(MovingQCS, self).__init__()
        self.nb_step = nb_step
        self.percentile = percentile
        self.scale_mode = scale_mode
        self.momentum = momentum
        if self.scale_mode == 'layer_wise':
            self.moving_thresh = torch.ones(1)
        else:
            raise NotImplementedError('Other mode is not implemented now')
        self.__inspection()

    def __inspection(self):
        if self.scale_mode not in ['channel_wise', 'neuron_wise', 'layer_wise']:
            raise NotImplementedError("Only mode 'channel_wise', 'neuron_wise' and 'layer_wise' are in the plan")
        assert self.momentum is None or 0 <= self.momentum <= 1, 'specified momentum should be in the range [0,1].'
        assert self.percentile is None or 0 <= self.percentile <= 1, 'specified percentile should be in the range [0,1].'

    def forward(self, x):
        thresh = 0
        if self.moving_thresh != x.device:
            self.moving_thresh = self.moving_thresh.to(x.device)
        if self.training:
            #  todo: deal with channel-wise and neuron-wise
            if self.percentile is not None:
                thresh = torch.quantile(x, self.percentile)
            else:
                thresh = x.max()
            qcs = QCS(thresh, self.nb_step)
            self.moving_thresh = self.momentum * self.moving_thresh + (1 - self.momentum) * thresh
        else:
            qcs = QCS(self.moving_thresh, self.nb_step)
        return qcs(x)


round_fn = RoundSurrogate.apply


class TrainableReLU(nn.Module):
    def __init__(self, alpha):
        super(TrainableReLU, self).__init__()
        self.alpha = nn.Parameter(alpha)
        # self.alpha = 1

    def forward(self, x):
        # todo: element-wised
        return torch.clamp(torch.relu(self.alpha * x + 0.5), max=1.0, min=0.0)


class InvTanh(nn.Module):
    def __init__(self, alpha: float = 1.0, learnable=True):
        super(InvTanh, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha])) if learnable else alpha

    def get_temperature(self):
        return self.alpha.detach().clone()

    def forward(self, x):
        return 0.5 * torch.tanh(self.alpha * x) + 0.5


class NoisySpike(nn.Module):
    # todo: suppport neuron-wised scale factor
    def __init__(self, p=1, sig='clamp'):
        super(NoisySpike, self).__init__()
        self.p = p
        if sig == 'clamp':
            self.sig = TrainableReLU(torch.tensor(1.0))
        elif sig == 'tanh':
            self.sig = InvTanh(alpha=2.5)

    def forward(self, x):
        if self.training:
            return self.sig(x) + ((x >= 0).float() - self.sig(x)).detach()
        return (x >= 0).float()


class NoisyQC(nn.Module):
    def __init__(self, thresh, nb_steps, noise_prob, detach=False, abs_if=True):
        super(NoisyQC, self).__init__()
        self.thresh = thresh if isinstance(thresh, nn.Parameter) else nn.Parameter(thresh)
        self.nb_steps = nb_steps
        self.noise_prob = noise_prob  # straight through with noise_prob
        self.clip = Clamp()
        self.detach = detach
        self.abs_if = abs_if

    def create_mask(self, x: torch.Tensor):
        return torch.bernoulli(torch.ones_like(x) * (1 - self.noise_prob))

    def forward(self, z):
        thresh = self.thresh
        if self.abs_if:
            thresh = torch.abs(self.thresh)
        # thresh = self.thresh
        if self.training:
            z = z / thresh * self.nb_steps
            if self.detach:
                z = z + self.create_mask(z) * (round_fn(z) - z).detach()
            else:
                z = z + self.create_mask(z) * (round_fn(z) - z)
            return thresh * self.clip(z / self.nb_steps)
        else:
            return thresh * self.clip(round_fn(z / thresh * self.nb_steps) / self.nb_steps)

    # def pclamp(self):
    #     self.thresh.clamp_(min=0.)


class GradFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


myfloor = GradFloor.apply


class MyFloor(nn.Module):
    def __init__(self, up=8., t=32):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t

    def forward(self, x):
        x = x / self.up
        x = myfloor(x * self.t + 0.5) / self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x
