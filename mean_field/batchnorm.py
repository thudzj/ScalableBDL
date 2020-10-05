import math

import torch
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F

class _BayesBatchNormMF(Module):
    r"""
    Applies Bayesian Batch Normalization over a 2D or 3D input

    Arguments:

    .. note:: other arguments are following batchnorm of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py

    """

    _version = 2
    __constants__ = ['track_running_stats',
                     'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, single_eps, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, deterministic=False):
        super(_BayesBatchNormMF, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.single_eps = single_eps
        self.deterministic = deterministic
        if self.affine:

            self.weight_mu = Parameter(torch.Tensor(num_features))
            self.weight_log_sigma = Parameter(torch.Tensor(num_features))

            self.bias_mu = Parameter(torch.Tensor(num_features))
            self.bias_log_sigma = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight_mu', None)
            self.register_parameter('weight_log_sigma', None)
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight_mu.data.uniform_()
            self.weight_log_sigma.data.fill_(-5)
            self.bias_mu.data.zero_()
            self.bias_log_sigma.data.fill_(-5)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        out = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

        if self.affine :
            if self.single_eps or self.deterministic:
                if self.deterministic:
                    weight = self.weight_mu
                    bias = self.bias_mu
                else:
                    weight = self.weight_mu + torch.exp(self.weight_log_sigma) * \
                            torch.randn(self.num_features, device=input.device, dtype=input.dtype)
                    bias = self.bias_mu + torch.exp(self.bias_log_sigma) * \
                            torch.randn(self.num_features, device=input.device, dtype=input.dtype)
                weight = weight.unsqueeze(0)
                bias = bias.unsqueeze(0)
            else:
                weight = self.weight_mu + torch.exp(self.weight_log_sigma) * \
                        torch.randn(input.shape[0], self.num_features, device=input.device, dtype=input.dtype)
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * \
                        torch.randn(input.shape[0], self.num_features, device=input.device, dtype=input.dtype)
            if out.dim() == 4:
                out = torch.addcmul(bias[:, :, None, None], weight[:, :, None, None], out)
            elif out.dim() == 2:
                out = torch.addcmul(bias, weight, out)
            else:
                raise NotImplementedError
        return out

    def extra_repr(self):
        return '{num_features}, ' \
                'eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BayesBatchNormMF, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class BayesBatchNorm2dMF(_BayesBatchNormMF):
    r"""
    Applies Bayesian Batch Normalization over a 2D input

    Arguments:

    .. note:: other arguments are following batchnorm of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py

    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class BayesBatchNorm1dMF(_BayesBatchNormMF):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
