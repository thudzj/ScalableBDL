import math

import torch
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F

class BayesLinearEMP(Module):
    r"""
    Applies Bayesian Linear
    """
    __constants__ = ['bias', 'in_features', 'out_features', 'num_mc_samples']

    def __init__(self, in_features, out_features, bias=True, num_mc_samples=20):
        super(BayesLinearEMP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mc_sample_id = None
        self.num_mc_samples = num_mc_samples

        self.parallel_eval = False

        self.weights = Parameter(torch.Tensor(num_mc_samples, out_features, in_features))

        if bias is None or bias is False:
            self.bias = False
        else:
            self.bias = True

        if self.bias:
            self.biases = Parameter(torch.Tensor(num_mc_samples, out_features))
        else:
            self.register_parameter('biases', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(2))
        for i in range(self.num_mc_samples):
            self.weights[i].data.uniform_(-stdv, stdv)
            if self.bias:
                self.biases[i].data.uniform_(-stdv, stdv)

    def forward(self, input):
        r"""
        Overriden.
        """
        if self.parallel_eval:
            if input.dim() == 2:
                input = input.unsqueeze(1).repeat(1, self.num_mc_samples, 1)
            out = torch.bmm(self.weights, input.permute(1, 2, 0)).permute(2, 0, 1)
            if self.bias:
                out = out + self.biases
        elif isinstance(self.mc_sample_id, int):
            weight = self.weights[self.mc_sample_id % self.num_mc_samples]
            bias = self.biases[self.mc_sample_id % self.num_mc_samples] if self.bias else None
            out = F.linear(input, weight, bias)
        else:
            bs = input.shape[0]
            idx = torch.tensor(self.mc_sample_id, device=input.device, dtype=torch.long)
            weight = self.weights[idx]
            out = torch.bmm(weight, input.unsqueeze(2)).squeeze()
            if self.bias:
                bias = self.biases[idx]
                out = out + bias
        return out

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'in_features={}, out_features={}, bias={}, num_mc_samples={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.num_mc_samples)
