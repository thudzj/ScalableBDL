import math

import torch
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F

class BayesLinearMF(Module):
    r"""
    Applies Bayesian Linear

    Arguments:

    .. note:: other arguments are following linear of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py

    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, single_eps, local_reparam, in_features, out_features, bias=True, deterministic=False):
        super(BayesLinearMF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.single_eps = single_eps
        self.local_reparam = local_reparam
        self.deterministic = deterministic

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))

        if bias is None or bias is False:
            self.bias = False
        else :
            self.bias = True

        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(-5)
        if self.bias :
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(-5)

    def forward(self, input):
        r"""
        Overriden.
        """
        if self.single_eps or self.deterministic:
            if self.deterministic:
                weight = self.weight_mu
                bias = self.bias_mu if self.bias else None
            else:
                weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn(self.out_features, self.in_features, device=input.device, dtype=input.dtype)
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn(self.out_features, device=input.device, dtype=input.dtype) if self.bias else None
            out = F.linear(input, weight, bias)
        else:
            if self.local_reparam:
                act_mu = F.linear(input, self.weight_mu, self.bias_mu if self.bias else None)
                act_var = F.linear(input**2, self.weight_log_sigma.exp()**2,
                                   self.bias_log_sigma.exp()**2 if self.bias else None)
                act_std = torch.sqrt(act_var+1e-16)
                out = act_mu + act_std * torch.randn_like(act_mu)
            else:
                weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn(input.shape[0], self.out_features, self.in_features, device=input.device, dtype=input.dtype)
                out = torch.bmm(weight, input.unsqueeze(2)).squeeze()
                if self.bias:
                    bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn(input.shape[0], self.out_features, device=input.device, dtype=input.dtype)
                    out = out + bias
        return out

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)
