import torch
import torch.nn as nn
import torch.nn.functional as F


class tdBatchNorm(nn.BatchNorm2d):
    """tdBN的实现。相关论文链接：https://arxiv.org/pdf/2011.05280。具体是在BN时，也在时间域上作平均；并且在最后的系数中引入了alpha变量以及Vth。
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, vth=0.5):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.vth = vth

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.vth * (input - mean[None, :, None, None, None]) / (
            torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return input


class spike_multiply_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike_a: torch.Tensor, spike_b: torch.Tensor):
        # y = spike_a * spike_b
        assert spike_a.shape == spike_b.shape, print('x.shape != spike.shape')  # 禁用广播机制
        if spike_a.dtype == torch.bool:
            spike_a_bool = spike_a
        else:
            spike_a_bool = spike_a.bool()

        if spike_b.dtype == torch.bool:
            spike_b_bool = spike_b
        else:
            spike_b_bool = spike_b.bool()

        if spike_a.dtype == torch.bool and spike_b.dtype == bool:
            # 若spike_a 和 spike_b 都是bool，则不应该需要计算梯度，因bool类型的tensor无法具有gard
            return spike_a_bool.logical_and(spike_b_bool)

        if spike_a.requires_grad and spike_b.requires_grad:
            ctx.save_for_backward(spike_b_bool, spike_a_bool)
        elif spike_a. \
                requires_grad and not spike_b.requires_grad:
            ctx.save_for_backward(spike_b_bool)
        elif not spike_a.requires_grad and spike_b.requires_grad:
            ctx.save_for_backward(spike_a_bool)

        ret = spike_a_bool.logical_and(spike_b_bool).float()
        ret.requires_grad_(spike_a.requires_grad or spike_b.requires_grad)
        return ret

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_spike_a = None
        grad_spike_b = None
        # grad_spike_a = grad_output * grad_spike_b
        # grad_spike_b = grad_output * grad_spike_a
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_spike_a = grad_output * ctx.saved_tensors[0]
            grad_spike_b = grad_output * ctx.saved_tensors[1]
        elif ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            grad_spike_a = grad_output * ctx.saved_tensors[0]
        elif not ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_spike_b = grad_output * ctx.saved_tensors[0]

        return grad_spike_a, grad_spike_b


class multiply_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, spike: torch.Tensor):
        # y = x * spike
        assert x.shape == spike.shape, print('x.shape != spike.shape')  # 禁用广播机制
        if spike.dtype == torch.bool:
            spike_bool = spike
        else:
            spike_bool = spike.bool()
        if x.requires_grad and spike.requires_grad:
            ctx.save_for_backward(spike_bool, x)
        elif x.requires_grad and not spike.requires_grad:
            ctx.save_for_backward(spike_bool)
        elif not x.requires_grad and spike.requires_grad:
            ctx.save_for_backward(x)
        return x * spike_bool

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_x = None
        grad_spike = None
        # grad_x = grad_output * spike
        # grad_spike = grad_output * x
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_x = grad_output * ctx.saved_tensors[0]
            grad_spike = grad_output * ctx.saved_tensors[1]
        elif ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            grad_x = grad_output * ctx.saved_tensors[0]
        elif not ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_spike = grad_output * ctx.saved_tensors[0]

        return grad_x, grad_spike


def mul(x: torch.Tensor, spike: torch.Tensor, spike_mul_spike=False):
    '''
    * :ref:`API in English <mul-en>`

    .. _mul-cn:

    :param x: 任意tensor
    :type x: torch.Tensor
    :param spike: 脉冲tensor。要求 ``spike`` 中的元素只能为 ``0`` 和 ``1``，或只为 ``False`` 和 ``True``，且 ``spike.shape`` 必须与 ``x.shape`` 相同
    :type spike: torch.Tensor
    :param spike_mul_spike: 是否为两个脉冲数据相乘，即``x`` 是否也是脉冲数据，即满足元素只能为 ``0`` 和 ``1``，或只为 ``False`` 和 ``True``。若 ``x`` 满足
        这一条件，则会调用更高级别的加速
    :type spike_mul_spike: bool
    :return: ``x * spike``
    :rtype: torch.Tensor

    针对与脉冲这一特殊的数据类型，进行前反向传播加速并保持数值稳定的乘法运算。

    * :ref:`中文API <mul-cn>`

    .. _mul-en:

    :param x: an arbitrary tensor
    :type x: torch.Tensor
    :param spike: a spike tensor. The elements in ``spike`` must be ``0`` and ``1`` or ``False`` and ``True``, and ``spike.shape`` should be same
        with ``x.shape``
    :type spike: torch.Tensor
    :param spike_mul_spike: whether spike multiplies spike, or whether ``x`` is the spiking data. When the elements in ``x`` are ``0`` and ``1`` or ``False`` and ``True``,
        this param can be ``True`` and this function will call an advanced accelerator
    :type spike_mul_spike: torch.Tensor
    :return: ``x * spike``
    :rtype: torch.Tensor

    Multiplication operation for an arbitrary tensor and a spike tensor, which is specially optimized for memory, speed, and
    numerical stability.
    '''
    if spike_mul_spike:
        return spike_multiply_spike.apply(x, spike)
    else:
        return multiply_spike.apply(x, spike)


class td_Dropout(nn.Module):
    def __init__(self, p=0.5, dropout_spikes=False):
        '''
        * :ref:`API in English <Dropout.__init__-en>`

        .. _Dropout.__init__-cn:

        :param p: 每个元素被设置为0的概率
        :type p: float
        :param dropout_spikes: 本层是否作用于脉冲数据，例如放在 ``neuron.LIFNode`` 层之后。若为 ``True``，则计算会有一定的加速
        :type dropout_spikes: bool

        与 ``torch.nn.Dropout`` 的几乎相同。区别在于，在每一轮的仿真中，被设置成0的位置不会发生改变；直到下一轮运行，即网络调用reset()函\\
        数后，才会按照概率去重新决定，哪些位置被置0。

        .. tip::
            这种Dropout最早由 `Enabling Spike-based Backpropagation for Training Deep Neural Network Architectures
            <https://arxiv.org/abs/1903.06379>`_ 一文进行详细论述：

            There is a subtle difference in the way dropout is applied in SNNs compared to ANNs. In ANNs, each epoch of
            training has several iterations of mini-batches. In each iteration, randomly selected units (with dropout ratio of :math:`p`)
            are disconnected from the network while weighting by its posterior probability (:math:`1-p`). However, in SNNs, each
            iteration has more than one forward propagation depending on the time length of the spike train. We back-propagate
            the output error and modify the network parameters only at the last time step. For dropout to be effective in
            our training method, it has to be ensured that the set of connected units within an iteration of mini-batch
            data is not changed, such that the neural network is constituted by the same random subset of units during
            each forward propagation within a single iteration. On the other hand, if the units are randomly connected at
            each time-step, the effect of dropout will be averaged out over the entire forward propagation time within an
            iteration. Then, the dropout effect would fade-out once the output error is propagated backward and the parameters
            are updated at the last time step. Therefore, we need to keep the set of randomly connected units for the entire
            time window within an iteration.

        * :ref:`中文API <Dropout.__init__-cn>`

        .. _Dropout.__init__-en:

        :param p: probability of an element to be zeroed
        :type p: float
        :param dropout_spikes: whether dropout is applied to spikes, such as after ``neuron.LIFNode``. If ``True``,
            the calculation will be accelerated
        :type dropout_spikes: bool

        This layer is almost same with ``torch.nn.Dropout``. The difference is that elements have been zeroed at first
        step during a simulation will always be zero. The indexes of zeroed elements will be update only after ``reset()``
        has been called and a new simulation is started.

        .. admonition:: Tip
            :class: tip

            This kind of Dropout is firstly described in `Enabling Spike-based Backpropagation for Training Deep Neural
            Network Architectures <https://arxiv.org/abs/1903.06379>`_:

            There is a subtle difference in the way dropout is applied in SNNs compared to ANNs. In ANNs, each epoch of
            training has several iterations of mini-batches. In each iteration, randomly selected units (with dropout ratio of :math:`p`)
            are disconnected from the network while weighting by its posterior probability (:math:`1-p`). However, in SNNs, each
            iteration has more than one forward propagation depending on the time length of the spike train. We back-propagate
            the output error and modify the network parameters only at the last time step. For dropout to be effective in
            our training method, it has to be ensured that the set of connected units within an iteration of mini-batch
            data is not changed, such that the neural network is constituted by the same random subset of units during
            each forward propagation within a single iteration. On the other hand, if the units are randomly connected at
            each time-step, the effect of dropout will be averaged out over the entire forward propagation time within an
            iteration. Then, the dropout effect would fade-out once the output error is propagated backward and the parameters
            are updated at the last time step. Therefore, we need to keep the set of randomly connected units for the entire
            time window within an iteration.
        '''
        super().__init__()
        assert 0 < p < 1
        self.mask = None
        self.p = p
        self.dropout_spikes = dropout_spikes

    def extra_repr(self):
        return 'p={}, dropout_spikes={}'.format(
            self.p, self.dropout_spikes
        )

    def create_mask(self, x: torch.Tensor):
        self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)

    def forward(self, x: torch.Tensor):
        if self.training:
            if self.mask is None:
                self.create_mask(x)
                # return x * self.mask
            if self.dropout_spikes:
                return mul(self.mask, x)
            else:
                return x * self.mask
        else:
            # return x / (1 - self.p)
            return x

    def reset(self):
        '''
        * :ref:`API in English <Dropout.reset-en>`

        .. _Dropout.reset-cn:

        :return: None

        本层是一个有状态的层。此函数重置本层的状态变量。

        * :ref:`中文API <Dropout.reset-cn>`

        .. _Dropout.reset-en:

        :return: None

        This layer is stateful. This function will reset all stateful variables.
        '''
        self.mask = None


class td_Dropout2(nn.Module):
    def __init__(self, p=0.5, dropout_spikes=False):
        super().__init__()
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
                return x.mul(self.mask)
        else:
            return x

    def reset(self):
        self.mask = None




# a = torch.rand((10,3,3))
# c2 = td_Dropout2(0.5,True)
# for i in range(2):
#     #c2 = td_Dropout(0.5)
#     #c2.reset()
#     for j in range(3):
#         print(c2(a[i, :, :]))
#     c2.reset()
