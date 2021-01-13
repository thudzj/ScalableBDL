# refer to https://github.com/Harry24k/pytorch-custom-utils/blob/master/torchhk/transform.py
import copy
import warnings

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d

from . import BayesLinearLR, BayesConv2dLR

def to_bayesian(input, num_mc_samples=20, rank=1, is_residual=False):

    if isinstance(input, (Linear, Conv2d)):
        if isinstance(input, (Linear)):
            output = BayesLinearLR(input.in_features, input.out_features,
                                   input.bias, num_mc_samples=num_mc_samples,
                                   rank=rank)
        elif isinstance(input, (Conv2d)):
            output = BayesConv2dLR(input.in_channels, input.out_channels,
                                   input.kernel_size, input.stride,
                                   input.padding, input.dilation,
                                   input.groups, input.bias,
                                   num_mc_samples=num_mc_samples,
                                   rank=rank)

        if input.weight is not None:
            if is_residual:
                output.weight.data = torch.eye(input.weight.data.size(
                    0)).unsqueeze(2).unsqueeze(3).float().data
            else:
                with torch.no_grad():
                    output.weight = input.weight

        if hasattr(input, 'bias') and input.bias is not None:
            if is_residual:
                output.bias.data.zero_()
            else:
                with torch.no_grad():
                    output.bias = input.bias
        del input
        return output

    output = input
    for name, module in input.named_children():
        output.add_module(name, to_bayesian(module, num_mc_samples, rank, is_residual))
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
            output.weight = input.weight
            output.bias = input.bias
        del input
        return output

    output = input
    for name, module in input.named_children():
        output.add_module(name, to_deterministic(module))
    del input
    return input
