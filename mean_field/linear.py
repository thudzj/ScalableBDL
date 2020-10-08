import math

import torch
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F

from .utils import MulExpAddFunction

class BayesLinearMF(Module):
    r"""
    Applies Bayesian Linear
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, 
                 deterministic=False):
        super(BayesLinearMF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.deterministic = deterministic

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_psi = Parameter(torch.Tensor(out_features, in_features))

        if bias is None or bias is False:
            self.bias = False
        else:
            self.bias = True

        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_psi = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_psi', None)

        self.reset_parameters()

        self.weight_size = list(self.weight_mu.shape)
        self.bias_size = list(self.bias_mu.shape) if self.bias else None
        self.mul_exp_add = MulExpAddFunction.apply

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_psi.data.uniform_(-6, -5)
        if self.bias:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_psi.data.uniform_(-6, -5)

    def forward(self, input):
        r"""
        Overriden.
        """
        if self.deterministic:
            weight = self.weight_mu
            bias = self.bias_mu if self.bias else None
            out = F.linear(input, weight, bias)
        else:
            bs = input.shape[0]
            weight = self.mul_exp_add(torch.randn(bs, *self.weight_size, 
                                                  device=input.device, 
                                                  dtype=input.dtype),
                                      self.weight_psi, self.weight_mu)

            out = torch.bmm(weight, input.unsqueeze(2)).squeeze()
            if self.bias:
                bias = self.mul_exp_add(torch.randn(bs, *self.bias_size, 
                                                  device=input.device, 
                                                  dtype=input.dtype),
                                        self.bias_psi, self.bias_mu)
                out = out + bias
        return out

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
