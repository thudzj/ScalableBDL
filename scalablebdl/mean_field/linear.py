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
                 deterministic=False, num_mc_samples=None):
        super(BayesLinearMF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.deterministic = deterministic
        self.num_mc_samples = num_mc_samples
        self.parallel_eval = False

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

        self.local_reparam = False
        self.flipout = False
        self.single_eps = False

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
        elif not self.parallel_eval:
            if self.single_eps:
                weight = torch.randn_like(self.weight_mu).mul_(self.weight_psi.exp()).add_(self.weight_mu)
                if self.bias:
                    bias = torch.randn_like(self.bias_mu).mul_(self.bias_psi.exp()).add_(self.bias_mu)
                else:
                    bias = None
                out = F.linear(input, weight, bias)
            elif self.local_reparam:
                act_mu = F.linear(input, self.weight_mu, None)
                act_var = F.linear(input**2, (self.weight_psi*2).exp_(), None)
                act_std = act_var.clamp(1e-8).sqrt_()
                out = torch.randn_like(act_mu).mul_(act_std).add_(act_mu)
                if self.bias:
                    bias = torch.randn(input.shape[0], *self.bias_size, device=input.device, dtype=input.dtype).mul_(self.bias_psi.exp()).add_(self.bias_mu)
                    out = out + bias
            elif self.flipout:
                outputs = F.linear(input, self.weight_mu, self.bias_mu if self.bias else None)
                # sampling perturbation signs
                sign_input = torch.empty_like(input).uniform_(-1, 1).sign()
                sign_output = torch.empty_like(outputs).uniform_(-1, 1).sign()
                # gettin perturbation weights
                delta_kernel = torch.randn_like(self.weight_psi).mul(self.weight_psi.exp())
                delta_bias = torch.randn_like(self.bias_psi).mul(self.bias_psi.exp()) if self.bias else None
                # perturbed feedforward
                perturbed_outputs = F.linear(input * sign_input, delta_kernel, delta_bias)
                out = outputs + perturbed_outputs * sign_output
            else:
                bs = input.shape[0]
                weight = self.mul_exp_add(torch.empty(bs, *self.weight_size,
                                                      device=input.device,
                                                      dtype=input.dtype).normal_(0, 1),
                                          self.weight_psi, self.weight_mu)

                out = torch.bmm(weight, input.unsqueeze(2)).squeeze()
                if self.bias:
                    bias = self.mul_exp_add(torch.empty(bs, *self.bias_size,
                                                      device=input.device,
                                                      dtype=input.dtype).normal_(0, 1),
                                            self.bias_psi, self.bias_mu)
                    out = out + bias
        else:
            if input.dim() == 2:
                input = input.unsqueeze(1).repeat(1, self.num_mc_samples, 1)
            weight = self.mul_exp_add(torch.empty(self.num_mc_samples,
                                                  *self.weight_size,
                                                  device=input.device,
                                                  dtype=input.dtype).normal_(0, 1),
                                      self.weight_psi, self.weight_mu)
            out = torch.bmm(weight, input.permute(1, 2, 0)).permute(2, 0, 1)
            if self.bias:
                bias = self.mul_exp_add(torch.empty(self.num_mc_samples,
                                                    *self.bias_size,
                                                    device=input.device,
                                                    dtype=input.dtype).normal_(0, 1),
                                        self.bias_psi, self.bias_mu)
                out = out + bias
        return out

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
