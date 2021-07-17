import math
import numpy as np

import torch
import torch.nn.init as init
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class _BayesConvNdLR(Module):
    r"""
    Applies Bayesian Convolution
    """
    __constants__ = ['stride', 'padding', 'dilation',
                     'groups', 'bias', 'in_channels',
                     'out_channels', 'kernel_size', 'num_mc_samples', 'rank']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias, num_mc_samples, rank, pert_init_std):
        super(_BayesConvNdLR, self).__init__()
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
        self.mc_sample_id = None
        self.deterministic = False
        self.num_mc_samples = num_mc_samples
        self.rank = rank
        self.pert_init_std = pert_init_std

        self.weight_mu = Parameter(torch.Tensor(out_channels,
                                in_channels // groups, *self.kernel_size))
        self.in_perturbations = Parameter(torch.Tensor(num_mc_samples, rank,
                                          in_channels//groups*np.prod(list( self.kernel_size))))
        self.out_perturbations = Parameter(torch.Tensor(num_mc_samples,
                                                        out_channels, rank))
        self.weight_size = list(self.weight_mu.shape)

        if bias is None or bias is False:
            self.register_parameter('bias', None)
        else:
            self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n *= np.prod(list(self.kernel_size))
        stdv = 1.0 / math.sqrt(n)
        self.weight_mu.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        m = math.sqrt(1./self.rank)
        v = math.sqrt((math.sqrt(self.rank*(self.pert_init_std**2)+1) - 1)/self.rank)
        self.in_perturbations.data.normal_(m, v)
        self.out_perturbations.data.normal_(m, v)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        s += ', padding={padding}'
        s += ', dilation={dilation}'
        s += ', groups={groups}'
        s += ', num_mc_samples={num_mc_samples}'
        s += ', rank={rank}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_BayesConvNdLR, self).__setstate__(state)

class BayesConv2dLR(_BayesConvNdLR):
    r"""
    Applies Bayesian Convolution for 2D inputs
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 num_mc_samples=20, rank=1, pert_init_std=0.2):
        super(BayesConv2dLR, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, num_mc_samples, rank, pert_init_std)
        self.parallel_eval = False

    def forward(self, input):
        r"""
        Overriden.
        """
        if self.deterministic:
            out = F.conv2d(input, weight=self.weight_mu,
                           bias=self.bias,
                           stride=self.stride, dilation=self.dilation,
                           groups=self.groups, padding=self.padding)
        elif self.parallel_eval:
            if input.dim() == 4:
                input = input.unsqueeze(1).repeat(1, self.num_mc_samples, 1, 1, 1)

            perturbations = torch.bmm(self.out_perturbations, self.in_perturbations)
            weight = perturbations.view(self.num_mc_samples, *self.weight_size).mul_(self.weight_mu)
            out = F.conv2d(input.flatten(start_dim=1, end_dim=2),
                           weight=weight.flatten(0, 1), bias=None,
                           stride=self.stride, dilation=self.dilation,
                           groups=self.groups*self.num_mc_samples,
                           padding=self.padding)
            out = out.view(out.shape[0], self.num_mc_samples,
                           self.out_channels, out.shape[2], out.shape[3])
            if self.bias is not None:
                out = out + self.bias[None, None, :, None, None]
        elif isinstance(self.mc_sample_id, int):
            self.mc_sample_id %= self.num_mc_samples
            perturbations = torch.matmul(self.out_perturbations[self.mc_sample_id],
                self.in_perturbations[self.mc_sample_id])
            weight = perturbations.view(*self.weight_size).mul_(self.weight_mu)
            out = F.conv2d(input, weight=weight,
                           bias=self.bias,
                           stride=self.stride, dilation=self.dilation,
                           groups=self.groups, padding=self.padding)
        else:
            bs = input.shape[0]
            idx = torch.tensor(self.mc_sample_id, device=input.device, dtype=torch.long)
            perturbations = torch.bmm(self.out_perturbations[idx], self.in_perturbations[idx])
            weight = perturbations.view(bs, *self.weight_size).mul_(self.weight_mu)
            out = F.conv2d(input.view(1, -1, input.shape[2], input.shape[3]),
                           weight=weight.flatten(0, 1), bias=None,
                           stride=self.stride, dilation=self.dilation,
                           groups=self.groups*bs, padding=self.padding)
            out = out.view(bs, self.out_channels, out.shape[2], out.shape[3])

            if self.bias is not None:
                out = out + bias[None, :, None, None]
        return out
