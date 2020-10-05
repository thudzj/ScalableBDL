""" Fisher information matrix approximations."""

from abc import ABC, abstractmethod
from typing import Union, List, Any

import torch
import torch.nn.functional as F
import tqdm

from .util import get_eigenvectors, kron


class Curvature(ABC):
    """Base class for all curvature approximations.

    All curvature approximations are computed layer-wise and stored in `state`.
    """

    def __init__(self, model: Union[torch.nn.Module, torch.nn.Sequential]) -> None:
        """Curvature class initializer.

        Args:
            model: This can be any (pre-trained) PyTorch model including all models from `torchvision`.
        """
        self.model = model
        self.state = dict()

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Abstract method to be implemented by each derived approximation individually."""
        pass


class DIAG(Curvature):
    """The diagonal Fisher information matrix approximation.

    Todo: Add equations.
    """

    def update(self, batch_size: int) -> None:
        """Computes the diagonal curvature for each `Conv2d` or `Linear` layer, skipping all others.

        Args:
            batch_size: The size of the current batch.
        """
        for module in self.model.modules():
            if module.__class__.__name__ in ['Linear', 'Conv2d']:
                grads = module.weight.grad.contiguous().view(module.weight.grad.shape[0], -1)
                if module.bias is not None:
                    grads = torch.cat([grads, module.bias.grad.unsqueeze(dim=1)], dim=1)
                grads = grads ** 2 * batch_size
                if module in self.state:
                    self.state[module] += grads
                else:
                    self.state[module] = grads


class FISHER(Curvature):
    """The Fisher information matrix approximation.

    Todo: Add equations.
    """
    def update(self, batch_size: int) -> None:
        """Computes the curvature for each `Conv2d` or `Linear` layer, skipping all others.

        Args:
            batch_size: The size of the current batch.
        """
        for module in self.model.modules():
            if module.__class__.__name__ in ['Linear', 'Conv2d']:
                grads = module.weight.grad.contiguous().view(module.weight.grad.shape[0], -1)
                if module.bias is not None:
                    grads = torch.cat([grads, module.bias.grad.contiguous().view(-1)])
                grads = torch.ger(grads, grads) * batch_size
                if module in self.state:
                    self.state[module] += grads
                else:
                    self.state[module] = grads


class KFAC(Curvature):
    """Computes and stores the Kronecker factored Fisher information matrix.

    Todo: Add source/equations.
    """
    def __init__(self, model: Union[torch.nn.Module, torch.nn.Sequential]):
        super().__init__(model)
        self.hooks = list()
        self.record = dict()

        for module in model.modules():
            if module.__class__.__name__ in ['Linear', 'Conv2d']:
                self.record[module] = [None, None]
                self.hooks.append(module.register_forward_pre_hook(self._save_input))
                self.hooks.append(module.register_backward_hook(self._save_output))

    def _save_input(self, module, input):
        self.record[module][0] = input[0]

    def _save_output(self, module, grad_input, grad_output):
        self.record[module][1] = grad_output[0] * grad_output[0].size(0)

    def update(self, batch_size: int) -> None:
        """Computes the first and second Kronecker factor
        for each `Conv2d` or `Linear` layer, skipping all others.

        Args:
            batch_size: The size of the current batch.
        """
        for module in self.model.modules():
            module_class = module.__class__.__name__
            if module_class in ['Linear', 'Conv2d']:
                forward, backward = self.record[module]

                # Compute first factor
                if module_class == 'Conv2d':
                    forward = F.unfold(forward, module.kernel_size, padding=module.padding, stride=module.stride)
                    forward = forward.data.permute(1, 0, 2).contiguous().view(forward.shape[1], -1)
                else:
                    forward = forward.data.t()
                if module.bias is not None:
                    ones = torch.ones_like(forward[:1])
                    forward = torch.cat([forward, ones], dim=0)
                first_factor = torch.mm(forward, forward.t()) / float(forward.shape[1])

                # Compute second factor
                if module_class == 'Conv2d':
                    backward = backward.data.permute(1, 0, 2, 3).contiguous().view(backward.shape[1], -1)
                else:
                    backward = backward.data.t()
                second_factor = torch.mm(backward, backward.t()) / float(backward.shape[1])

                if module in self.state:
                    self.state[module][0] += first_factor
                    self.state[module][1] += second_factor
                else:
                    self.state[module] = [first_factor, second_factor]


class EFB(Curvature):
    """Computes the eigenvalue corrected Kronecker factored Fisher information matrix.

    Todo: Add source/equations.
    """
    def __init__(self, model: Union[torch.nn.Module, torch.nn.Sequential],
                 factors: List[torch.Tensor]) -> None:
        super().__init__(model)
        self.eigvecs = get_eigenvectors(factors)
        self.diags = dict()

    def update(self, batch_size: int) -> None:
        """Computes the eigenvalue corrected diagonal of the Fisher information matrix.

        Args:
            batch_size: The size of the current batch.
        """
        layer = 0
        for module in self.model.modules():
            if module.__class__.__name__ in ['Linear', 'Conv2d']:
                grads = module.weight.grad.contiguous().view(module.weight.grad.shape[0], -1)
                if module.bias is not None:
                    grads = torch.cat([grads, module.bias.grad.unsqueeze(dim=1)], dim=1)
                lambdas = (self.eigvecs[layer][1].t() @ grads @ self.eigvecs[layer][0]) ** 2

                if module in self.state:
                    self.state[module] += lambdas
                    self.diags[module] += grads ** 2 * batch_size
                else:
                    self.state[module] = lambdas
                    self.diags[module] = grads ** 2 * batch_size
                layer += 1


class INF:
    """Computes the diagonal correction term and low-rank approximations of KFAC factor eigenvectors and EFB diagonals.

    Todo: Add more info from paper.
    """
    def __init__(self, factors: List[torch.Tensor],
                 lambdas: List[torch.Tensor],
                 diags: List[torch.Tensor]) -> None:
        self.eigvecs = get_eigenvectors(factors)
        self.lambdas = lambdas
        self.diags = diags
        self.state = list()

    def accumulate(self, rank=100):
        """Accumulates the diagonal values used for the diagonal correction term.

        Todo: Add more info from paper.
        Args:
            rank: The rank of the low-rank approximations.
        """
        for eigvecs, lambdas, diags in tqdm.tqdm(zip(self.eigvecs, self.lambdas, self.diags), total=len(self.eigvecs)):
            xxt_eigvecs, ggt_eigvecs = eigvecs
            lambda_vec = lambdas.t().contiguous().view(-1)
            diag_vec = diags.t().contiguous().view(-1)

            lr_xxt_eigvecs, lr_ggt_eigvecs, lr_lambda = self._dim_reduction(xxt_eigvecs, ggt_eigvecs, lambda_vec, rank)
            sif_diag = self._diagonal_accumulator(lr_xxt_eigvecs, lr_ggt_eigvecs, lr_lambda)

            self.state.append([lr_xxt_eigvecs, lr_ggt_eigvecs, lr_lambda, diag_vec - sif_diag])

    @staticmethod
    def _dim_reduction(xxt_eigvecs, ggt_eigvecs, lambda_vec, rank):
        if rank >= lambda_vec.shape[0]:
            return xxt_eigvecs, ggt_eigvecs, lambda_vec
        else:
            m = ggt_eigvecs.shape[1]
            idx_total_0 = torch.argsort(-torch.abs(lambda_vec))
            idx_total = idx_total_0 + 1
            idx_top_l = idx_total[0:rank]
            idx_left = list()
            idx_right = list()
            for z in range(0, rank):
                i = int((idx_top_l[z] - 1) / m) + 1
                j = idx_top_l[z] - (m * (i - 1))
                idx_left.append(i)
                idx_right.append(j)

            idx_top_lm = list()
            idx_left = torch.unique(torch.tensor(idx_left))
            idx_right = torch.unique(torch.tensor(idx_right))
            len_l = len(idx_left)
            len_r = len(idx_right)

            for i in range(0, len_l):
                for j in range(0, len_r):
                    idx_top_lm.append(m * (idx_left[i] - 1) + idx_right[j])

            lr_lambda = lambda_vec[[idx - 1 for idx in idx_top_lm]]
            lr_cov_inner = xxt_eigvecs[:, [idx - 1 for idx in idx_left]]
            lr_cov_outer = ggt_eigvecs[:, [idx - 1 for idx in idx_right]]

            return lr_cov_inner, lr_cov_outer, lr_lambda

    @staticmethod
    def _diagonal_accumulator(xxt_eigvecs, ggt_eigvecs, lambda_vec):
        n = xxt_eigvecs.shape[0]
        m = ggt_eigvecs.shape[0]
        diag_vec = torch.zeros(n * m).to(lambda_vec.device)
        k = 0

        for i in range(n):
            diag_kron = kron(xxt_eigvecs[i, :].unsqueeze(0), ggt_eigvecs) ** 2
            diag_vec[k:k + m] = diag_kron @ lambda_vec
            k += m
        return diag_vec
