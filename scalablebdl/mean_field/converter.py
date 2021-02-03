# refer to https://github.com/Harry24k/pytorch-custom-utils/blob/master/torchhk/transform.py
import copy
import warnings

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU

from . import BayesLinearMF, BayesConv2dMF, BayesBatchNorm2dMF, BayesPReLUMF

def to_bayesian(input, psi_init_range=[-6, -5], num_mc_samples=20):

    if isinstance(input, (Linear, Conv2d, BatchNorm2d, PReLU)):
        if isinstance(input, (Linear)):
            output = BayesLinearMF(input.in_features, input.out_features,
                                   input.bias, num_mc_samples=num_mc_samples)
        elif isinstance(input, (Conv2d)):
            output = BayesConv2dMF(input.in_channels, input.out_channels,
                                   input.kernel_size, input.stride,
                                   input.padding, input.dilation,
                                   input.groups, input.bias,
                                   num_mc_samples=num_mc_samples)
        elif isinstance(input, (PReLU)):
            output = BayesPReLUMF(input.num_parameters, num_mc_samples=num_mc_samples)
        else:
            output = BayesBatchNorm2dMF(input.num_features, input.eps,
                                        input.momentum, input.affine,
                                        input.track_running_stats,
                                        num_mc_samples=num_mc_samples)
            output.running_mean = input.running_mean
            output.running_var = input.running_var
            output.num_batches_tracked = input.num_batches_tracked

        if input.weight is not None:
            with torch.no_grad():
                output.weight_mu = input.weight

        if hasattr(input, 'bias') and input.bias is not None:
            with torch.no_grad():
                output.bias_mu = input.bias

        if output.weight_psi is not None:
            output.weight_psi.data.uniform_(psi_init_range[0], psi_init_range[1])
        if hasattr(output, 'bias_psi') and output.bias_psi is not None:
            output.bias_psi.data.uniform_(psi_init_range[0], psi_init_range[1])
        del input
        return output

    output = input
    for name, module in input.named_children():
        output.add_module(name, to_bayesian(module, psi_init_range, num_mc_samples))
    del input
    return output

def to_deterministic(input):

    if isinstance(input, (BayesLinearMF, BayesConv2dMF, BayesBatchNorm2dMF)):
        if isinstance(input, (BayesLinearMF)):
            output = Linear(input.in_features, input.out_features, input.bias)
        elif isinstance(input, (BayesConv2dMF)):
            output = Conv2d(input.in_channels, input.out_channels,
                            input.kernel_size, input.stride,
                            input.padding, input.dilation,
                            input.groups, input.bias)
        elif isinstance(input, (BayesPReLUMF)):
            output = PReLU(input.num_parameters)
        else:
            output = BatchNorm2d(input.num_features, input.eps,
                                 input.momentum, input.affine,
                                 input.track_running_stats)
            output.running_mean = input.running_mean
            output.running_var = input.running_var
            output.num_batches_tracked = input.num_batches_tracked

        with torch.no_grad():
            if input.weight is not None:
                output.weight = input.weight_mu
            if hasattr(input, 'bias') and input.bias is not None:
                output.bias = input.bias_mu
        del input
        return output
    output = input
    for name, module in input.named_children():
        output.add_module(name, to_deterministic(module))
    del input
    return output
