import math

import torch
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F

class _BayesBatchNormEMP(Module):
    r"""
    Applies Bayesian Batch Normalization over a 2D or 3D input
    """
    __constants__ = ['track_running_stats',
                     'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine', 'num_mc_samples']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, num_mc_samples=20):
        super(_BayesBatchNormEMP, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.mc_sample_id = None
        self.num_mc_samples = num_mc_samples
        self.parallel_eval = False
        if self.affine:
            self.weights = Parameter(torch.Tensor(num_mc_samples, num_features))
            self.biases = Parameter(torch.Tensor(num_mc_samples, num_features))
        else:
            self.register_parameter('weights', None)
            self.register_parameter('biases', None)
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
            self.weights.data.fill_(1)
            self.biases.data.zero_()

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

        if self.parallel_eval:
            if input.dim() == 4:
                input = input.unsqueeze(1).repeat(1, self.num_mc_samples, 1, 1, 1)
            elif input.dim() == 2:
                input = input.unsqueeze(1).repeat(1, self.num_mc_samples, 1)
            input = input.flatten(start_dim=0, end_dim=1)
        out = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

        if self.affine :
            if self.parallel_eval:
                if out.dim() == 4:
                    out = out.view(-1, self.num_mc_samples, out.shape[1],
                        out.shape[2], out.shape[3]) * self.weights[None, :, :, None, None] \
                        + self.biases[None, :, :, None, None]
                elif out.dim() == 2:
                    out = out.view(-1, self.num_mc_samples, out.shape[1]) \
                        * self.weights[None, :, :] + self.biases[None, :, :]
                else:
                    raise NotImplementedError

            elif isinstance(self.mc_sample_id, int):
                self.mc_sample_id = self.mc_sample_id % self.num_mc_samples
                weight = self.weights[self.mc_sample_id:(self.mc_sample_id+1)]
                bias = self.biases[self.mc_sample_id:(self.mc_sample_id+1)]

                if out.dim() == 4:
                    out = torch.addcmul(bias[:, :, None, None],
                                        weight[:, :, None, None], out)
                elif out.dim() == 2:
                    out = torch.addcmul(bias, weight, out)
                else:
                    raise NotImplementedError
            else:
                bs = input.shape[0]
                idx = torch.tensor(self.mc_sample_id, device=input.device, dtype=torch.long)
                weight = self.weights[idx]
                bias = self.biases[idx]

                if out.dim() == 4:
                    out = torch.addcmul(bias[:, :, None, None],
                                        weight[:, :, None, None], out)
                elif out.dim() == 2:
                    out = torch.addcmul(bias, weight, out)
                else:
                    raise NotImplementedError


        return out

    def extra_repr(self):
        return '{num_features}, {num_mc_samples},' \
                'eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BayesBatchNormEMP, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class BayesBatchNorm2dEMP(_BayesBatchNormEMP):
    r"""
    Applies Bayesian Batch Normalization over a 2D input
    """
    def _check_input_dim(self, input):
        if input.dim() != 4 and input.dim() != 5:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class BayesBatchNorm1dEMP(_BayesBatchNormEMP):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
