# refer to https://github.com/Harry24k/pytorch-custom-utils/blob/master/torchhk/transform.py
import copy
import warnings

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d

from . import BayesLinearIMP, BayesConv2dIMP, BayesBatchNorm2dIMP

def to_bayesian(input):
    return _to_bayesian(copy.deepcopy(input))

def _to_bayesian(input):

    if isinstance(input, (Linear, Conv2d, BatchNorm2d)):
        if isinstance(input, (Linear)):
            output = BayesLinearIMP(input.in_features, input.out_features, input.bias)
        elif isinstance(input, (Conv2d)):
            output = BayesConv2dIMP(input.in_channels, input.out_channels,
                                   input.kernel_size, input.stride,
                                   input.padding, input.dilation,
                                   input.groups, input.bias)
        else:
            output = BayesBatchNorm2dIMP(input.num_features, input.eps,
                                        input.momentum, input.affine,
                                        input.track_running_stats)
            setattr(output, 'running_mean', getattr(input, 'running_mean'))
            setattr(output, 'running_var', getattr(input, 'running_var'))
            setattr(output, 'num_batches_tracked', getattr(input, 'num_batches_tracked'))
        # todo
        setattr(output, 'weight_mu', getattr(input, 'weight'))
        setattr(output, 'bias_mu', getattr(input, 'bias'))

        return output
    else:
        for name, module in input.named_children():
            setattr(input, name, to_bayesian(module))
        return input

def to_deterministic(input):
    return _to_deterministic(copy.deepcopy(input))

def _to_deterministic(input):

    if isinstance(input, (BayesLinearIMP, BayesConv2dIMP, BayesBatchNorm2dIMP)):
        if isinstance(input, (BayesLinearIMP)):
            output = Linear(input.in_features, input.out_features, input.bias)
        elif isinstance(input, (BayesConv2dIMP)):
            output = Conv2d(input.in_channels, input.out_channels,
                            input.kernel_size, input.stride,
                            input.padding, input.dilation,
                            input.groups, input.bias)
        else:
            output = BatchNorm2d(input.num_features, input.eps,
                                 input.momentum, input.affine,
                                 input.track_running_stats)
            setattr(output, 'running_mean', getattr(input, 'running_mean'))
            setattr(output, 'running_var', getattr(input, 'running_var'))
            setattr(output, 'num_batches_tracked', getattr(input, 'num_batches_tracked'))
        # todo
        setattr(output, 'weight', getattr(input, 'weight_mu'))
        setattr(output, 'bias', getattr(input, 'bias_mu'))
        return output
    else:
        for name, module in input.named_children():
            setattr(input, name, to_deterministic(module))
        return input
