"""This module contains all utility functions apart from plotting which are in `plot.py`."""

import argparse
import multiprocessing
import os
from typing import Tuple, List, Any, Union

import numpy as np
import psutil
import torch
import tqdm
from scipy.stats import entropy


def get_eigenvalues(factors: List[torch.Tensor],
                    verbose=False) -> torch.Tensor:
    """Computes the eigenvalues of KFAC, EFB or diagonal factors.

    Args:
        factors: A list of KFAC, EFB or diagonal factors.
        verbose (optional): Prints out progress if True. Defaults to False.

    Returns:
        The eigenvalues of all KFAC, EFB or diagonal factors.
    """
    eigenvalues = torch.Tensor()
    factors = tqdm.tqdm(factors, disable=not verbose)
    for layer, factor in enumerate(factors):
        factors.set_description(desc=f"Layer [{layer + 1}/{len(factors)}]")
        if len(factor) == 2:
            xxt_eigvals = torch.symeig(factor[0])[0]
            ggt_eigvals = torch.symeig(factor[1])[0]
            eigenvalues = torch.cat([eigenvalues, torch.ger(xxt_eigvals, ggt_eigvals).contiguous().view(-1)])
        else:
            eigenvalues = torch.cat([eigenvalues, factor.contiguous().view(-1)])
    return eigenvalues


def get_eigenvectors(factors: List[torch.Tensor]) -> List[List[torch.Tensor]]:
    """Computes the eigenvectors of KFAC factors.

    Args:
        factors: A dictionary of KFAC factors.

    Returns:
        A list where each element is a tuple of first and second KFAC factors eigenvectors.
    """
    eigenvectors = list()
    for (xxt, ggt) in factors:
        sym_xxt, sym_ggt = xxt + xxt.t(), ggt + ggt.t()
        _, xxt_eigvecs = torch.symeig(sym_xxt, eigenvectors=True)
        _, ggt_eigvecs = torch.symeig(sym_ggt, eigenvectors=True)
        eigenvectors.append([xxt_eigvecs, ggt_eigvecs])
    return eigenvectors


def linear_interpolation(min_val: float,
                         max_val: float,
                         data: np.ndarray) -> np.ndarray:
    """Performs a linear interpolation of `data` between `min_val` and `max_val`.

    Args:
        min_val: The lower bound of the interpolation.
        max_val: The upper bound of the interpolation.
        data: The data to be interpolated.

    Returns:
        The linearly interpolated data.
    """
    return (max_val - min_val) * (data - np.min(data)) / (np.max(data) - np.min(data)) + min_val


def accuracy(probabilities: np.ndarray,
             labels: np.ndarray) -> float:
    """Computes the top 1 accuracy of the predicted class probabilities in percent.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.

    Returns:
        The top 1 accuracy in percent.
    """
    return 100.0 * np.mean(np.argmax(probabilities, axis=1) == labels)


def binned_kl_distance(dist1: np.ndarray,
                       dist2: np.ndarray,
                       smooth=1e-7,
                       bins=np.logspace(-7, 1, num=200)) -> float:
    """Computes the symmetric, discrete Kulback-Leibler divergence (JSD) between two distributions.

    Todo: Add source.

    Args:
        dist1: The first distribution.
        dist2: The second distribution.
        smooth (optional): Smoothing factor to prevent numerical instability.
        bins (optional): How to discretize the distributions.

    Returns:
        The JSD.
    """
    dist1_pdf, _ = np.histogram(dist1, bins)
    dist2_pdf, _ = np.histogram(dist2, bins)

    dist1_pdf = dist1_pdf + smooth
    dist2_pdf = dist2_pdf + smooth

    dist1_pdf_normalized = dist1_pdf / dist1_pdf.sum()
    dist2_pdf_normalized = dist2_pdf / dist2_pdf.sum()

    dir1_normalized_entropy = entropy(dist1_pdf_normalized, dist2_pdf_normalized)
    dir2_normalized_entropy = entropy(dist2_pdf_normalized, dist1_pdf_normalized)

    return dir1_normalized_entropy + dir2_normalized_entropy


def confidence(probabilities: np.ndarray,
               mean=True) -> Union[float, np.ndarray]:
    """The confidence of a prediction is the maximum of the predicted class probabilities.

    Args:
        probabilities: The predicted class probabilities.
        mean (optional): If True, returns the average confidence over all provided predictions. Defaults to True.

    Returns:
        The confidence.
    """
    if mean:
        return np.mean(np.max(probabilities, axis=1))
    return np.max(probabilities, axis=1)


def negative_log_likelihood(probabilities: np.ndarray,
                            labels: np.ndarray) -> float:
    """Computes the Negative Log-Likelihood (NLL) of the predicted class probabilities.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.

    Returns:
        The NLL.
    """
    return -np.mean(np.log(probabilities[np.arange(probabilities.shape[0]), labels] + 1e-12))


def calibration_curve(probabilities: np.ndarray,
                      labels: np.ndarray,
                      bins=20) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the Expected Calibration Error (ECE) of the predicted class probabilities.

    Todo: Add explanation and/or source/formula.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        bins (optional): The number of bins into which the probabilities are discretized. Defaults to 20.

    Returns:
        The ECE alongside the average confidence, accuracy and proportion of data points in each bin respectively.
    """
    confidences = np.max(probabilities, 1)
    step = (confidences.shape[0] + bins - 1) // bins
    bins = np.sort(confidences)[::step]
    if confidences.shape[0] % step != 1:
        bins = np.concatenate((bins, [np.max(confidences)]))
    # bins = np.linspace(0.1, 1.0, 30)
    predictions = np.argmax(probabilities, 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    accuracies = predictions == labels

    xs = []
    ys = []
    zs = []

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            xs.append(avg_confidence_in_bin)
            ys.append(accuracy_in_bin)
            zs.append(prop_in_bin)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    return ece, xs, ys, zs


def expected_calibration_error(probabilities: np.ndarray,
                               labels: np.ndarray,
                               bins=10) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the Expected Calibration Error (ECE) of the predicted class probabilities.

    Todo: Add explanation and/or source/formula.

    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        bins (optional): The number of bins into which the probabilities are discretized. Defaults to 10.

    Returns:
        The ECE alongside the average confidence, accuracy and proportion of data points in each bin respectively.
    """
    conf = confidence(probabilities, mean=False)
    edges = np.linspace(0, 1, bins + 1)
    bin_ace = list()
    bin_accuracy = list()
    bin_confidence = list()
    ece = 0
    for i in range(bins):
        mask = np.logical_and(conf > edges[i], conf <= edges[i + 1])
        if any(mask):
            bin_acc = accuracy(probabilities[mask], labels[mask]) / 100
            bin_conf = conf[mask].mean()
            ace = bin_conf - bin_acc
            ece += mask.mean() * np.abs(ace)

            bin_ace.append(ace)
            bin_accuracy.append(bin_acc)
            bin_confidence.append(bin_conf)
        else:
            bin_ace.append(0)
            bin_accuracy.append(0)
            bin_confidence.append(0)
    return ece, np.array(bin_ace), np.array(bin_accuracy), np.array(bin_confidence)


def predictive_entropy(probabilities: np.ndarray,
                       mean=False) -> Union[np.ndarray, float]:
    """Computes the predictive entropy of the predicted class probabilities.

    Todo: Add formula.

    Args:
        probabilities: The predicted class probabilities.
        mean (optional): If True, returns the average predictive entropy over all provided predictions.
        Defaults to False.

    Returns:
        The predictive entropy.
    """
    pred_ent = np.apply_along_axis(entropy, axis=1, arr=probabilities)
    if mean:
        return np.mean(pred_ent)
    return pred_ent


def ram() -> float:
    """Returns the total amount of utilized system memory (RAM) in percent.

    Returns:
        RAM usage in percent.
    """
    return psutil.virtual_memory()[2]


def vram() -> float:
    """Determines the amount of video memory (VRAM) utilized by the current process in GB.

    Returns:
        VRAM usage in GB.
    """
    return torch.cuda.memory_allocated() / (1024.0 ** 3)


def kron(a: torch.Tensor,
         b: torch.Tensor) -> torch.Tensor:
    """Computes the Kronecker product between the two 2D-matrices (tensors) `a` and `b`.

    Todo: Add example.

    Args:
        a: A 2D-matrix (tensor)
        b: A 2D-matrix (tensor)

    Returns:
        The Kronecker product between `a` and `b`.
    """
    return torch.einsum("ab,cd->acbd", [a, b]).contiguous().view(a.size(0) * b.size(0), a.size(1) * b.size(1))


def setup() -> Any:
    """Initializes values of importance for most modules and parses command line arguments.

    Returns:
        The parsed arguments.
    """
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    gpu = torch.cuda.is_available()
    if gpu:
        capability = torch.cuda.get_device_capability()
        capable = capability[0] + 0.1 * capability[1] >= 3.5
        device = torch.device('cuda') if capable else torch.device('cpu')
    else:
        device = torch.device('cpu')
    cpus = multiprocessing.cpu_count()
    root_dir = "/home/matthias/Data/Ubuntu"
    torch_dir = "/home/matthias/.torch"
    results_dir = "/home/matthias/Data/Ubuntu/results"

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=device, type=type(device), help="Computation device: GPU or CPU")
    parser.add_argument("--torch_dir", default=torch_dir, type=str, required=False,
                        help="Path to torchvision modelzoo location")
    parser.add_argument("--torch_data", default=os.path.join(torch_dir, "datasets"), type=str,
                        help="Path to torchvision datasets location")
    parser.add_argument("--root_dir", default=root_dir, type=str, help="Path to root dir")
    parser.add_argument("--data_dir", default=os.path.join(root_dir, "datasets"), type=str,
                        required=False, help="Path to datasets location")
    parser.add_argument("--results_dir", default=results_dir, type=str, required=False, help="Path to results dir")

    parser.add_argument("--mode", default="torch", type=str, help="GPU/PyTorch or CPU computation (default: Torch)")
    parser.add_argument("--parallel", action="store_true", help="Use data parallelism (default: off)")
    parser.add_argument("--ram", default=psutil.virtual_memory()[1] / 1023 ** 3,
                        help="Amount of available system memory on process start.")
    parser.add_argument("--cpus", default=cpus, type=int, help="Number of CPUs (default: Auto)")
    parser.add_argument("--workers", default=cpus - 1, type=int,
                        help="Data loading workers (default: cpus - 1)")
    parser.add_argument("--prefix", default="", type=str, help="Filename prefix (default: None)")
    parser.add_argument("--suffix", default="", type=str, help="Filename suffix (default: None)")

    parser.add_argument("--model", default=None, type=str, required=True, help="Name of model to use (default: None)")
    parser.add_argument("--data", default="imagenet", type=str, help="Name of dataset (default: imagenet)")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size (default: 32)")
    parser.add_argument("--epochs", default=1, type=int, help="Number of (training) epochs (default: 1)")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate in SGD training (default: 1e-3)")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum in SGD training (default: 0.9)")
    parser.add_argument("--l2", default=0, type=float, help="L2-norm regularization strength (default: 0)")
    parser.add_argument("--optimizer", default="random", type=str,
                        help="Optimizer used for hyperparameter optimization (deafult: random)")

    parser.add_argument("--estimator", default="kfac", type=str, help="Fisher estimator (default: kfac)")
    parser.add_argument("--samples", default=30, type=int, help="Number of posterior weight samples (default: 30)")
    parser.add_argument("--calls", default=50, type=int, help="Number of hyperparameter search calls (default: 50)")
    parser.add_argument("--boundaries", action="store_true",
                        help="Whether the to search the hyperparameter space boundaries (default: off)")
    parser.add_argument("--exp_id", default=-1, type=str, help="Experiment ID (default: -1)")
    parser.add_argument("--layer", action="store_true", help="Whether layer-wise damping should be used. (default: off)")
    parser.add_argument("--pre_scale", default=1, type=int,
                        help="Size of dataset, multiplied by scaling factor (default: 1)")
    parser.add_argument("--augment", action="store_true", help="Whether to use data augmentation (default: off)")
    parser.add_argument("--norm", default=-1, type=float,
                        help="This times identity is added to Kronecker factors (default: -1)")
    parser.add_argument("--scale", default=-1, type=float,
                        help="Kronecker factors are multiplied by this times pre-scale (default: -1)")
    parser.add_argument("--epsilon", default=0, type=float, help="Step size for FGSM (default: 0)")
    parser.add_argument("--rank", default=100, type=int, help="Rank for information form sparsification (default: 100)")

    parser.add_argument("--plot", action="store_true",
                        help="Whether to plot the evaluation results or not (default: off)")
    parser.add_argument("--no_results", action="store_true",
                        help="Whether to not save the evaluation results (default: off)")
    parser.add_argument("--stats", action="store_true", help="Whether to compute running statistics (default: off)")
    parser.add_argument("--calibration", action="store_true", help="Make calibration plots (default: off)")
    parser.add_argument("--ood", action="store_true", help="Run ood evaluation/make ood plots (default: off)")
    parser.add_argument("--fgsm", action="store_true", help="Run FGSM evaluation/make fgsm plots (default: off)")
    parser.add_argument("--ecdf", action="store_true", help="Plot inverse ECDF vs. predictive entropy (default: off)")
    parser.add_argument("--entropy", action="store_true", help="Plot predictive entropy histogram (default: off)")
    parser.add_argument("--summary", action="store_true", help="Print a model summary (default: off)")
    parser.add_argument("--eigvals", action="store_true", help="Plot eigenvalue histogram (default: off)")
    parser.add_argument("--hyper", action="store_true", help="Plot hyperparameter optimization results (default: off)")
    parser.add_argument("--networks", action="store_true", help="Plot network calibration comparison")
    parser.add_argument("--verbose", action="store_true", help="Give verbose output during execution. (default: off)")

    args = parser.parse_args()
    os.environ['TORCH_HOME'] = args.torch_dir
    return args
