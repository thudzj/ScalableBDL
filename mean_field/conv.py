import math

import torch
import torch.nn.init as init
from torch.nn import Module, Parameter
import torch.nn.functional as F
import numpy as np

from torch.nn.modules.utils import _pair

def stat_cuda(msg):
    print('--', msg)
    print('allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_cached() / 1024 / 1024,
        torch.cuda.max_memory_cached() / 1024 / 1024
    ))

class MulExpAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.mark_dirty(input)
        output = input.mul_(weight.exp()).add_(bias)
        ctx.save_for_backward(bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        bias, output = ctx.saved_tensors
        grad_weight = (grad_output*(output - bias)).sum(0)
        grad_bias = grad_output.sum(0)
        return None, grad_weight, grad_bias

class _BayesConvNdMF(Module):
    r"""
    Applies Bayesian Convolution

    Arguments:

    .. note:: other arguments are following conv of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
    """
    __constants__ = ['stride', 'padding', 'dilation',
                     'groups', 'bias', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias):
        super(_BayesConvNdMF, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight_mu = Parameter(torch.Tensor(
            out_channels, in_channels // groups, kernel_size, kernel_size))
        self.weight_log_sigma = Parameter(torch.Tensor(
            out_channels, in_channels // groups, kernel_size, kernel_size))

        if bias is None or bias is False :
            self.bias = False
        else :
            self.bias = True

        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_log_sigma = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n *= self.kernel_size ** 2
        stdv = 1.0 / math.sqrt(n)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(-5)

        if self.bias :
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(-5)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is False:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_BayesConvNdMF, self).__setstate__(state)

class BayesConv2dMF(_BayesConvNdMF):
    r"""
    Applies Bayesian Convolution for 2D inputs

    Arguments:

    .. note:: other arguments are following conv of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py

    """
    def __init__(self, single_eps, local_reparam, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, deterministic=False):
        super(BayesConv2dMF, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.single_eps = single_eps
        self.local_reparam = local_reparam
        assert(not self.bias)
        # assert(self.groups == 1)
        self.deterministic = deterministic
        self.weight_size = list(self.weight_mu.shape)
        self.mul_exp_add = MulExpAddFunction.apply

    def forward(self, input):
        r"""
        Overriden.
        """
        if self.single_eps or self.deterministic:
            if self.deterministic:
                weight = self.weight_mu
            else:
                weight = torch.randn(*self.weight_size, device=input.device, dtype=input.dtype).mul(
                                        self.weight_log_sigma.exp()).add(self.weight_mu)
            out = F.conv2d(input, weight=weight, bias=None,
                            stride=self.stride, dilation=self.dilation,
                            groups=self.groups, padding=self.padding)
        else:
            if self.local_reparam:
                act_mu = F.conv2d(input, weight=self.weight_mu, bias=None,
                                  stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, groups=self.groups)
                act_var = F.conv2d(input**2,
                                  weight=(self.weight_log_sigma*2).exp_(), bias=None,
                                  stride=self.stride, padding=self.padding,
                                  dilation=self.dilation, groups=self.groups)#.sqrt_()
                act_std = torch.sqrt(act_var+1e-10)
                out =  torch.randn_like(act_mu).mul_(act_std).add_(act_mu)
            else:
                bs = input.shape[0]
                weight = self.mul_exp_add(torch.randn(bs, *self.weight_size, device=input.device, dtype=input.dtype),
                                    self.weight_log_sigma, self.weight_mu).view(
                                    bs*self.weight_size[0], *self.weight_size[1:])
                out = F.conv2d(input.view(1, -1, input.shape[2], input.shape[3]), weight=weight, bias=None,
                                stride=self.stride, dilation=self.dilation,
                                groups=self.groups*bs, padding=self.padding)
                out = out.view(bs, self.out_channels, out.shape[2], out.shape[3])
        return out
