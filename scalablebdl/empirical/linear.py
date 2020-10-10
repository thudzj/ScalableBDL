import math

import torch
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F

class BayesLinearEMP(Module):
    r"""
    Applies Bayesian Linear
    """
    __constants__ = ['bias', 'in_features', 'out_features', 'num_modes']

    def __init__(self, in_features, out_features, bias=True, num_modes=20):
        super(BayesLinearEMP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = None
        self.num_modes = num_modes

        self.weights = Parameter(torch.Tensor(num_modes, out_features, in_features))

        if bias is None or bias is False:
            self.bias = False
        else:
            self.bias = True

        if self.bias:
            self.biases = Parameter(torch.Tensor(num_modes, out_features))
        else:
            self.register_parameter('biases', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weights.size(2))
        for i in range(self.num_modes):
            self.weights[i].data.uniform_(-stdv, stdv)
            if self.bias:
                self.biases[i].data.uniform_(-stdv, stdv)

    def forward(self, input):
        r"""
        Overriden.
        """
        if isinstance(self.mode, int):
            weight = self.weights[self.mode % self.num_modes]
            bias = self.biases[self.mode % self.num_modes] if self.bias else None
            out = F.linear(input, weight, bias)
        else:
            bs = input.shape[0]
            idx = torch.tensor(self.mode, device=input.device, dtype=torch.long)
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
        return 'in_features={}, out_features={}, bias={}, num_modes={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.num_modes)
