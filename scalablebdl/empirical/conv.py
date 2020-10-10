import math
import numpy as np

import torch
import torch.nn.init as init
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class _BayesConvNdEMP(Module):
    r"""
    Applies Bayesian Convolution
    """
    __constants__ = ['stride', 'padding', 'dilation',
                     'groups', 'bias', 'in_channels',
                     'out_channels', 'kernel_size', 'num_modes']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, num_modes):
        super(_BayesConvNdEMP, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.mode = None
        self.num_modes = num_modes

        self.weights = Parameter(torch.Tensor(
            num_modes, out_channels, in_channels // groups, *self.kernel_size))
        self.weight_size = list(self.weights.shape)[1:]

        if bias is None or bias is False:
            self.bias = False
            self.register_parameter('biases', None)
        else:
            self.bias = True
            self.biases = Parameter(torch.Tensor(num_modes, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n *= np.prod(list(self.kernel_size))
        stdv = 1.0 / math.sqrt(n)
        for i in range(self.num_modes):
            self.weights[i].data.uniform_(-stdv, stdv)
            if self.bias :
                self.biases[i].data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        s += ', padding={padding}'
        s += ', dilation={dilation}'
        s += ', groups={groups}'
        s += ', bias={bias}'
        s += ', num_modes={num_modes}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_BayesConvNdEMP, self).__setstate__(state)

class BayesConv2dEMP(_BayesConvNdEMP):
    r"""
    Applies Bayesian Convolution for 2D inputs
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, num_modes=20):
        super(BayesConv2dEMP, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, num_modes)

    def forward(self, input):
        r"""
        Overriden.
        """
        if isinstance(self.mode, int):
            out = F.conv2d(input, weight=self.weights[self.mode % self.num_modes],
                           bias=self.biases[self.mode % self.num_modes] if self.bias else None,
                           stride=self.stride, dilation=self.dilation,
                           groups=self.groups, padding=self.padding)
        else:
            bs = input.shape[0]
            idx = torch.tensor(self.mode, device=input.device, dtype=torch.long)
            weight = self.weights[idx].view(bs*self.weight_size[0], *self.weight_size[1:])
            out = F.conv2d(input.view(1, -1, input.shape[2], input.shape[3]),
                           weight=weight, bias=None,
                           stride=self.stride, dilation=self.dilation,
                           groups=self.groups*bs, padding=self.padding)
            out = out.view(bs, self.out_channels, out.shape[2], out.shape[3])

            if self.bias:
                bias = self.biases[idx]
                out = out + bias[:, :, None, None]
        return out
