# refer to https://github.com/Harry24k/pytorch-custom-utils/blob/master/torchhk/transform.py
import copy
import warnings

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU

from . import BayesLinearEMP, BayesConv2dEMP, BayesBatchNorm2dEMP, BayesPReLUEMP

def to_bayesian(input, num_mc_samples=20, is_residual=False):
    return _to_bayesian(copy.deepcopy(input), num_mc_samples, is_residual)

def _to_bayesian(input, num_mc_samples=20, is_residual=False):

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
            setattr(output, 'running_mean', getattr(input, 'running_mean'))
            setattr(output, 'running_var', getattr(input, 'running_var'))
            setattr(output, 'num_batches_tracked', getattr(input, 'num_batches_tracked'))

        if input.weight is not None:
            if is_residual:
                if isinstance(input, (Conv2d)):
                    output.weights.data = torch.eye(input.weight.data.size(
                        0)).unsqueeze(2).unsqueeze(3).float().unsqueeze(0).repeat(
                            num_mc_samples, 1, 1, 1, 1).data
                elif isinstance(input, BatchNorm2d):
                    output.weights.data.fill_(1.)
            else:
                output.weights.data = input.weight.unsqueeze(0).repeat(
                    num_mc_samples, *([1,]*input.weight.dim())).data
        if hasattr(input, 'bias') and input.bias is not None:
            if is_residual:
                output.biases.data.zero_()
            else:
                output.biases.data = input.bias.unsqueeze(0).repeat(
                    num_mc_samples, *([1,]*input.bias.dim())).data

        return output
    else:
        for name, module in input.named_children():
            setattr(input, name, _to_bayesian(module, num_mc_samples, is_residual))
        return input

def to_deterministic(input):
    assert False, "Cannot convert an empirical BNN into DNN"
