from typing import List, Union

import numpy as np
import torch

from .util import kron


def invert_factors(factors: List[torch.Tensor],
                   norm: Union[float, list],
                   scale: Union[float, list],
                   estimator='kfac') -> List[torch.Tensor]:
    """Regularizes and inverts diagonal, KFAC, EFB or INF factors for sampling.

    Args:
        factors: The diagonal, KFAC, EFB or INF Fisher information matrix (FIM) for each layer.
        norm: This quantity times the identity is added to each factor (tau).
        scale: Each factor is multiplied by this quantity.
        estimator: The FIM estimator. One of `diag`, `kfac`, `efb` or `inf`.

    Returns:
        A list of inverted factors and potentially other quantities required for sampling.
    """
    inv_factors = list()
    for index, factor in enumerate(factors):
        if estimator == "kfac":
            frst, scnd = factor

            diag_frst = torch.diag(frst.new(frst.shape[0]).fill_(norm ** 0.5))
            diag_scnd = torch.diag(scnd.new(scnd.shape[0]).fill_(norm ** 0.5))

            reg_frst = scale ** 0.5 * frst + diag_frst
            reg_scnd = scale ** 0.5 * scnd + diag_scnd

            reg_frst = (reg_frst + reg_frst.t()) / 2.0
            reg_scnd = (reg_scnd + reg_scnd.t()) / 2.0

            try:
                chol_ifrst = reg_frst.inverse().cholesky()
                chol_iscnd = reg_scnd.inverse().cholesky()
            except RuntimeError:
                print("PyTorch Cholesky is singular. Using Numpy.")
                chol_ifrst = torch.from_numpy(np.linalg.cholesky(np.linalg.inv(reg_frst.cpu().numpy()))).to(frst.device)
                chol_iscnd = torch.from_numpy(np.linalg.cholesky(np.linalg.inv(reg_scnd.cpu().numpy()))).to(scnd.device)

            inv_factors.append((chol_ifrst, chol_iscnd))
        elif estimator == "efb":
            frst_eigvecs, scnd_eigvecs, lambda_ = factor
            reg_inv_lambda = torch.reciprocal(scale * lambda_ + norm).sqrt()
            inv_factors.append((frst_eigvecs, scnd_eigvecs, reg_inv_lambda))
        elif estimator == "inf":
            # Low-rank U_A, low-rank U_G, low-rank Lambda, D
            lr_frst_eigvecs, lr_scnd_eigvecs, lr_lambda, correction = factor
            correction[correction < 0] = 0

            reg_lr_lambda = (scale * lr_lambda).sqrt()
            reg_inv_correction = torch.reciprocal(scale * correction + norm).sqrt()

            pre_sample = pre_sampler(lr_frst_eigvecs, lr_scnd_eigvecs, reg_lr_lambda, reg_inv_correction)

            inv_factors.append((lr_frst_eigvecs, lr_scnd_eigvecs, reg_inv_correction, pre_sample))
        elif estimator == 'diag':
            if isinstance(norm, float) and isinstance(scale, float):
                inv_factors.append(torch.reciprocal(scale * factor + norm).sqrt())
            else:
                inv_factors.append(torch.reciprocal(scale[index] * factor + norm[index]).sqrt())
    return inv_factors


def sample_and_replace_weights(model: Union[torch.nn.Sequential, torch.nn.Module],
                               inv_factors: List[torch.Tensor],
                               estimator='kfac') -> None:
    """Samples a new set of weights from the approximate weight posterior distribution and replaces the existing ones.

    Args:
        model: A (pre-trained) PyTorch model.
        inv_factors: The inverted factors (plus further quantities required for sampling)
        estimator: The FIM estimator. One of `diag`, `kfac`, `efb` or `inf`.
    """

    index = 0
    for module in model.modules():
        if module.__class__.__name__ in ['Linear', 'Conv2d']:
            weight = module.weight
            bias = module.bias
            if estimator in ["kfac", "efb"]:
                if estimator == "kfac":
                    a, b = inv_factors[index]  # a: first KFAC factor, b: second KFAC factor
                else:
                    a, b, scale = inv_factors[index]  # a, b: Eigenvectors of first and second KFAC factor

                z = torch.randn(a.size(0), b.size(0), device=a.device, dtype=a.dtype)
                if estimator == "efb":
                    z *= scale.t()
                x = (a @ z @ b.t()).t()  # Final transpose because PyTorch uses channels first

            elif estimator == "diag":
                var = inv_factors[index]
                x = var.new(var.size()).normal_() * var

            elif estimator == "inf":
                a, b, c, d = inv_factors[index]
                x = sampler(a, b, c, d).reshape(a.shape[0], b.shape[0]).t()

            index += 1
            if bias is not None:
                bias_sample = x[:, -1].contiguous().view(*bias.shape)
                x = x[:, :-1]
                bias.data.add_(bias_sample)
            weight.data.add_(x.contiguous().view(*weight.shape))


def pre_sampler(frst_eigvecs: torch.Tensor,
                scnd_eigvecs: torch.Tensor,
                reg_lambda: torch.Tensor,
                reg_inv_correction: torch.Tensor) -> torch.Tensor:
    """Pre-sampler for INF sampling. Only needs to be called once.

    Args:
        frst_eigvecs: Eigenvectors of first KFAC factor.
        scnd_eigvecs: Eigenvectors of second KFAC factor.
        reg_lambda: Regularized, eigenvalue corrected diagonal FIM (computed by EFB)
        reg_inv_correction: Regularized inverse of the diagonal correction term of INF.

    Returns:
        A pre-sample used in `sampler` to sample weight sets.
    """
    scale_sqrt = torch.diag(reg_lambda)
    try:
        V_s = reg_inv_correction.contiguous().view(-1, 1) * kron(frst_eigvecs, scnd_eigvecs) @ scale_sqrt
    except RuntimeError:
        print("GPU capacity exhausted. Pre-sampling on CPU.")
        frst_eigvecs = frst_eigvecs.cpu()
        scnd_eigvecs = scnd_eigvecs.cpu()
        reg_inv_correction = reg_inv_correction.cpu()
        scale_sqrt = scale_sqrt.cpu()
        V_s = reg_inv_correction.contiguous().view(-1, 1) * kron(frst_eigvecs, scnd_eigvecs) @ scale_sqrt
    vtv = V_s.t() @ V_s
    vtv = (vtv + vtv.t()) / 2.0
    A_c_inv = vtv.cholesky().inverse()
    B_c = (vtv + torch.eye(scale_sqrt.shape[0], device=scale_sqrt.device)).cholesky()
    C = A_c_inv.t() @ (B_c - torch.eye(scale_sqrt.shape[0], device=scale_sqrt.device)) @ A_c_inv
    L_c = (C.inverse() + vtv).inverse()
    P_c = scale_sqrt @ L_c @ scale_sqrt

    return P_c.to(reg_lambda.device)


def sampler(frst_eigvecs: torch.Tensor,
            scnd_eigvecs: torch.Tensor,
            reg_inv_correction: torch.Tensor,
            pre_sample: torch.Tensor) -> torch.Tensor:
    """Samples a new set of weights from the INF weight posterior distribution for the current layer.

    Args:
        frst_eigvecs: Eigenvectors of first KFAC factor.
        scnd_eigvecs: Eigenvectors of second KFAC factor.
        reg_inv_correction: Regularized inverse of the diagonal correction term of INF.
        pre_sample: Pre-sample computed by the pre-sampler.

    Returns:
        A new set of weights for the current layer.
    """
    X = torch.randn(frst_eigvecs.shape[0] * scnd_eigvecs.shape[0], device=frst_eigvecs.device, dtype=frst_eigvecs.dtype)
    Y_l = reg_inv_correction * X
    unvec_Y_l = Y_l.t().reshape((scnd_eigvecs.shape[0], frst_eigvecs.shape[0]))
    Xq = scnd_eigvecs.t() @ unvec_Y_l @ frst_eigvecs
    Qx = pre_sample @ Xq.t().contiguous().view(-1)
    unvec_Qx = Qx.t().reshape((scnd_eigvecs.shape[1], frst_eigvecs.shape[1]))
    X_p_s = scnd_eigvecs @ unvec_Qx @ frst_eigvecs.t()
    Y_r = reg_inv_correction ** 2 * X_p_s.t().contiguous().view(-1)

    return Y_l.t() - Y_r.t()
