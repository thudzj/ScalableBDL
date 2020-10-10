# refer to https://github.com/Harry24k/pytorch-custom-utils/blob/master/torchhk/transform.py
import copy
import warnings

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d

from . import BayesLinearEMP, BayesConv2dEMP, BayesBatchNorm2dEMP

def to_bayesian(input, num_modes=20):
    return _to_bayesian(copy.deepcopy(input), num_modes)

def _to_bayesian(input, num_modes):

    if isinstance(input, (Linear, Conv2d, BatchNorm2d)):
        if isinstance(input, (Linear)):
            output = BayesLinearEMP(input.in_features, input.out_features,
                                    input.bias, num_modes=num_modes)
        elif isinstance(input, (Conv2d)):
            output = BayesConv2dEMP(input.in_channels, input.out_channels,
                                    input.kernel_size, input.stride,
                                    input.padding, input.dilation,
                                    input.groups, input.bias,
                                    num_modes=num_modes)
        else:
            output = BayesBatchNorm2dEMP(input.num_features, input.eps,
                                         input.momentum, input.affine,
                                         input.track_running_stats,
                                         num_modes=num_modes)
            setattr(output, 'running_mean', getattr(input, 'running_mean'))
            setattr(output, 'running_var', getattr(input, 'running_var'))
            setattr(output, 'num_batches_tracked', getattr(input, 'num_batches_tracked'))

        output.weights.data = input.weight.unsqueeze(0).repeat(
            num_modes, *([1,]*input.weight.dim())).data
        if input.bias is not None:
            output.biases.data = input.bias.unsqueeze(0).repeat(
                num_modes, *([1,]*input.bias.dim())).data

        return output
    else:
        for name, module in input.named_children():
            setattr(input, name, to_bayesian(module))
        return input

def to_deterministic(input):
    assert False, "Cannot convert an empirical BNN into DNN"
