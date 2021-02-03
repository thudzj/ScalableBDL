# refer to https://github.com/Harry24k/pytorch-custom-utils/blob/master/torchhk/transform.py
import copy
import warnings

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU

from . import BayesLinearEMP, BayesConv2dEMP, BayesBatchNorm2dEMP, BayesPReLUEMP

def to_bayesian(input, num_mc_samples=20):

    if isinstance(input, (Linear, Conv2d, BatchNorm2d, PReLU)):
        if isinstance(input, (Linear)):
            output = BayesLinearEMP(input.in_features, input.out_features,
                                    input.bias, num_mc_samples=num_mc_samples)
        elif isinstance(input, (Conv2d)):
            output = BayesConv2dEMP(input.in_channels, input.out_channels,
                                    input.kernel_size, input.stride,
                                    input.padding, input.dilation,
                                    input.groups, input.bias,
                                    num_mc_samples=num_mc_samples)
        elif isinstance(input, (PReLU)):
            output = BayesPReLUEMP(input.num_parameters, num_mc_samples=num_mc_samples)
        else:
            output = BayesBatchNorm2dEMP(input.num_features, input.eps,
                                         input.momentum, input.affine,
                                         input.track_running_stats,
                                         num_mc_samples=num_mc_samples)
            output.running_mean = input.running_mean
            output.running_var = input.running_var
            output.num_batches_tracked = input.num_batches_tracked

        if input.weight is not None:
            output.weights.data = input.weight.unsqueeze(0).repeat(
                num_mc_samples, *([1,]*input.weight.dim())).data
        if hasattr(input, 'bias') and input.bias is not None:
            output.biases.data = input.bias.unsqueeze(0).repeat(
                num_mc_samples, *([1,]*input.bias.dim())).data
        del input
        return output
    output = input
    for name, module in input.named_children():
        output.add_module(name, to_bayesian(module, num_mc_samples))
    del input
    return output

def to_deterministic(input):
    assert False, "Cannot convert an empirical BNN into DNN"
