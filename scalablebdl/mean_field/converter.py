# refer to https://github.com/Harry24k/pytorch-custom-utils/blob/master/torchhk/transform.py
import copy
import warnings

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU

from . import BayesLinearMF, BayesConv2dMF, BayesBatchNorm2dMF, BayesPReLUMF

def to_bayesian(input, psi_init_range=[-6, -5], num_mc_samples=20, is_residual=False):
    return _to_bayesian(copy.deepcopy(input), psi_init_range, num_mc_samples, is_residual)

def _to_bayesian(input, psi_init_range=[-6, -5], num_mc_samples=20, is_residual=False):

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
            setattr(output, 'running_mean', getattr(input, 'running_mean'))
            setattr(output, 'running_var', getattr(input, 'running_var'))
            setattr(output, 'num_batches_tracked', getattr(input, 'num_batches_tracked'))

        setattr(output, 'weight_mu', getattr(input, 'weight'))
        if hasattr(input, 'bias'):
            setattr(output, 'bias_mu', getattr(input, 'bias'))
        if is_residual:
            if isinstance(input, (Conv2d)):
                output.weight_mu.data = torch.eye(output.weight_mu.data.size(0)).unsqueeze(2).unsqueeze(3).float()
                if output.bias_mu is not None:
                    output.bias_mu.data.zero_()
            elif isinstance(input, BatchNorm2d):
                if output.affine:
                    output.weight_mu.data.fill_(1.)
                    output.bias_mu.data.zero_()

        if output.weight_psi is not None:
            output.weight_psi.data.uniform_(psi_init_range[0], psi_init_range[1])
            output.weight_psi.data = output.weight_psi.data.to(output.weight_mu.device)
        if hasattr(output, 'bias_psi') and output.bias_psi is not None:
            output.bias_psi.data.uniform_(psi_init_range[0], psi_init_range[1])
            output.bias_psi.data = output.bias_psi.data.to(output.bias_mu.device)

        return output
    else:
        for name, module in input.named_children():
            setattr(input, name, _to_bayesian(module, psi_init_range, num_mc_samples, is_residual))
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
