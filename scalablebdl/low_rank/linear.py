import math

import torch
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F

class BayesLinearLR(Module):
    r"""
    Applies Bayesian Linear
    """
    __constants__ = ['bias', 'in_features', 'out_features', 'num_mc_samples', 'rank']

    def __init__(self, in_features, out_features, bias=True, num_mc_samples=20, rank=1):
        super(BayesLinearLR, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mc_sample_id = None
        self.num_mc_samples = num_mc_samples
        self.rank = rank

        self.parallel_eval = False

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.in_perturbations = Parameter(torch.Tensor(num_mc_samples, rank, in_features))
        self.out_perturbations = Parameter(torch.Tensor(num_mc_samples, out_features, rank))

        if bias is None or bias is False:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.in_perturbations.data.fill_(math.sqrt(1./self.rank))
        self.out_perturbations.data.fill_(math.sqrt(1./self.rank))

    def forward(self, input):
        r"""
        Overriden.
        """
        if self.parallel_eval:
            if input.dim() == 2:
                input = input.unsqueeze(1).repeat(1, self.num_mc_samples, 1)
            perturbations = torch.bmm(self.out_perturbations, self.in_perturbations)
            weight = self.weight * perturbations
            out = torch.bmm(weight, input.permute(1, 2, 0)).permute(2, 0, 1)
            if self.bias is not None:
                out = out + self.bias
        elif isinstance(self.mc_sample_id, int):
            self.mc_sample_id %= self.num_mc_samples
            perturbations = torch.matmul(self.out_perturbations[self.mc_sample_id],
                self.in_perturbations[self.mc_sample_id])
            weight = self.weight * perturbations
            out = F.linear(input, weight, self.bias)
        else:
            bs = input.shape[0]
            idx = torch.tensor(self.mc_sample_id, device=input.device, dtype=torch.long)
            perturbations = torch.bmm(self.out_perturbations[idx], self.in_perturbations[idx])
            weight = self.weight * perturbations
            out = torch.bmm(weight, input.unsqueeze(2)).squeeze()
            if self.bias is not None:
                out = out + self.bias
        return out

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'in_features={}, out_features={}, bias={}, num_mc_samples={}, rank={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.num_mc_samples, self.rank)
