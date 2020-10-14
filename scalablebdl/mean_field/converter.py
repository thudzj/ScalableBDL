# refer to https://github.com/Harry24k/pytorch-custom-utils/blob/master/torchhk/transform.py
import copy
import warnings

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d

from . import BayesLinearMF, BayesConv2dMF, BayesBatchNorm2dMF

def to_bayesian(input, psi_init_range=[-6, -5]):
    return _to_bayesian(copy.deepcopy(input), psi_init_range)

def _to_bayesian(input, psi_init_range=[-6, -5]):

    if isinstance(input, (Linear, Conv2d, BatchNorm2d)):
        if isinstance(input, (Linear)):
            output = BayesLinearMF(input.in_features, input.out_features, input.bias)
        elif isinstance(input, (Conv2d)):
            output = BayesConv2dMF(input.in_channels, input.out_channels,
                                   input.kernel_size, input.stride,
                                   input.padding, input.dilation,
                                   input.groups, input.bias)
        else:
            output = BayesBatchNorm2dMF(input.num_features, input.eps,
                                        input.momentum, input.affine,
                                        input.track_running_stats)
            setattr(output, 'running_mean', getattr(input, 'running_mean'))
            setattr(output, 'running_var', getattr(input, 'running_var'))
            setattr(output, 'num_batches_tracked', getattr(input, 'num_batches_tracked'))

        setattr(output, 'weight_mu', getattr(input, 'weight'))
        setattr(output, 'bias_mu', getattr(input, 'bias'))

        output.weight_psi.data.uniform_(psi_init_range[0], psi_init_range[1])
        output.weight_psi.data = output.weight_psi.data.to(output.weight_mu.device)
        if output.bias_psi is not None:
            output.bias_psi.data.uniform_(psi_init_range[0], psi_init_range[1])
            output.bias_psi.data = output.bias_psi.data.to(output.bias_mu.device)

        return output
    else:
        for name, module in input.named_children():
            setattr(input, name, _to_bayesian(module, psi_init_range))
        return input

def to_deterministic(input):
    return _to_deterministic(copy.deepcopy(input))

def _to_deterministic(input):

    if isinstance(input, (BayesLinearMF, BayesConv2dMF, BayesBatchNorm2dMF)):
        if isinstance(input, (BayesLinearMF)):
            output = Linear(input.in_features, input.out_features, input.bias)
        elif isinstance(input, (BayesConv2dMF)):
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

        setattr(output, 'weight', getattr(input, 'weight_mu'))
        setattr(output, 'bias', getattr(input, 'bias_mu'))
        return output
    else:
        for name, module in input.named_children():
            setattr(input, name, _to_deterministic(module))
        return input
