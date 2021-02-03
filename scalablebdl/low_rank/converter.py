# refer to https://github.com/Harry24k/pytorch-custom-utils/blob/master/torchhk/transform.py
import copy
import warnings

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d

from . import BayesLinearLR, BayesConv2dLR

def to_bayesian(input, rank=1, num_mc_samples=20, pert_init_std=0.2):

    if isinstance(input, (Linear, Conv2d)):
        if isinstance(input, (Linear)):
            output = BayesLinearLR(input.in_features, input.out_features,
                                   input.bias, num_mc_samples=num_mc_samples,
                                   rank=rank, pert_init_std=pert_init_std)
        elif isinstance(input, (Conv2d)):
            output = BayesConv2dLR(input.in_channels, input.out_channels,
                                   input.kernel_size, input.stride,
                                   input.padding, input.dilation,
                                   input.groups, input.bias,
                                   num_mc_samples=num_mc_samples,
                                   rank=rank, pert_init_std=pert_init_std)

        if input.weight is not None:
            with torch.no_grad():
                output.weight_mu = input.weight

        if hasattr(input, 'bias') and input.bias is not None:
            with torch.no_grad():
                output.bias = input.bias
        del input
        return output

    output = input
    for name, module in input.named_children():
        output.add_module(name, to_bayesian(module, rank, num_mc_samples, pert_init_std))
    del input
    return output

def to_deterministic(input):

    if isinstance(input, (BayesLinearLR, BayesConv2dLR)):
        if isinstance(input, (BayesLinearLR)):
            output = Linear(input.in_features, input.out_features, input.bias)
        elif isinstance(input, (BayesConv2dLR)):
            output = Conv2d(input.in_channels, input.out_channels,
                            input.kernel_size, input.stride,
                            input.padding, input.dilation,
                            input.groups, input.bias)

        with torch.no_grad():
            output.weight = input.weight_mu
            output.bias = input.bias
        del input
        return output

    output = input
    for name, module in input.named_children():
        output.add_module(name, to_deterministic(module))
    del input
    return output
