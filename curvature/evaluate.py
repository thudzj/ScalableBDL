import copy
import os

import numpy as np
import tabulate
import torch
import torchvision
import tqdm
from matplotlib import pyplot as plt

import datasets
from lenet5 import lenet5
from resnet import resnet18
import plot
from sampling import sample_and_replace_weights, invert_factors
from utils import (accuracy, setup, ram, vram, expected_calibration_error, predictive_entropy, negative_log_likelihood,
                   calibration_curve, get_eigenvectors)


def eval_fgsm(model,
              data,
              epsilon=0.1,
              stats=True,
              device=torch.device('cuda'),
              verbose=True):

    model.eval()
    logits_list = torch.Tensor().to(device)
    labels_list = torch.LongTensor()
    stats_dict = None

    data = tqdm.tqdm(data, disable=not verbose or len(data) == 1)
    for images, labels in data:
        data.set_postfix({'RAM': ram(), 'VRAM': vram()})

        adv_images = datasets.fgsm(model, images.to(device, non_blocking=True), labels.to(device, non_blocking=True),
                                   epsilon=epsilon)
        with torch.no_grad():
            adv_logits = model(adv_images)

        logits_list = torch.cat([logits_list, adv_logits])
        labels_list = torch.cat([labels_list, labels])

    adv_predictions = torch.nn.functional.softmax(logits_list, dim=1).detach().cpu().numpy()
    labels = labels_list.numpy()

    if stats:
        acc = accuracy(adv_predictions, labels)
        ece1 = 100 * expected_calibration_error(adv_predictions, labels)[0]
        ece2 = 100 * calibration_curve(adv_predictions, labels)[0]
        nll = negative_log_likelihood(adv_predictions, labels)
        ent = predictive_entropy(adv_predictions, mean=True)
        stats_dict = {"eps": epsilon, "acc": acc, "ece1": ece1, "ece2": ece2, "nll": nll, "ent": ent}

    if verbose:
        print(f"Step: {epsilon:.2f} | Adv. Entropy: {stats_dict['ent']:.2f} | Adv. Accuracy: {stats_dict['acc']:.2f}%")

    return adv_predictions, labels, stats_dict


def eval_fgsm_bnn(model,
                  data,
                  inv_factors,
                  estimator='kfac',
                  samples=30,
                  epsilon=0.1,
                  stats=True,
                  device=torch.device('cuda'),
                  verbose=True):

    model.eval()
    mean_state = copy.deepcopy(model.state_dict())
    mean_predictions = 0

    samples = tqdm.tqdm(range(samples), disable=not verbose)
    for sample in samples:
        samples.set_postfix({'RAM': ram(), 'VRAM': vram()})
        sample_and_replace_weights(model, inv_factors, estimator)
        predictions, labels, _ = eval_fgsm(model, data, epsilon, stats=False, device=device, verbose=False)
        mean_predictions += predictions
        model.load_state_dict(mean_state)
    mean_predictions /= len(samples)

    if stats:
        acc = accuracy(mean_predictions, labels)
        ece1 = 100 * expected_calibration_error(mean_predictions, labels)[0]
        ece2 = 100 * calibration_curve(mean_predictions, labels)[0]
        nll = negative_log_likelihood(mean_predictions, labels)
        ent = predictive_entropy(mean_predictions, mean=True)
        stats_dict = {"eps": epsilon, "acc": acc, "ece1": ece1, "ece2": ece2, "nll": nll, "ent": ent}

    if verbose:
        print(f"Step: {epsilon:.2f} | Adv. Entropy: {stats_dict['ent']:.2f} | Adv. Accuracy: {stats_dict['acc']:.2f}%")

    return mean_predictions, labels, stats_dict


def eval_nn(model, dataset, device=torch.device('cuda'), verbose=False):
    model.eval()

    with torch.no_grad():
        logits_list = torch.Tensor().to(device)
        labels_list = torch.LongTensor()

        dataset = tqdm.tqdm(dataset, disable=not verbose or len(dataset) == 1)
        for images, labels in dataset:
            dataset.set_postfix({'RAM': ram(), 'VRAM': vram()})

            logits = model(images.to(device, non_blocking=True))
            logits_list = torch.cat([logits_list, logits])
            labels_list = torch.cat([labels_list, labels])

        predictions = torch.nn.functional.softmax(logits_list, dim=1).cpu().numpy()
        labels = labels_list.numpy()

    if verbose:
        print(f"Accuracy: {accuracy(predictions, labels):.2f}% | ECE: {100 * expected_calibration_error(predictions, labels)[0]:.2f}%")

    return predictions, labels


def eval_bnn(model,
             dataset,
             inv_factors,
             estimator='kfac',
             samples=30, stats=False,
             device=torch.device('cuda'),
             verbose=True):

    model.eval()
    mean_state = copy.deepcopy(model.state_dict())
    mean_predictions = 0
    stats_list = {"acc": [], "ece": [], "nll": [], "ent": []}

    with torch.no_grad():
        samples = tqdm.tqdm(range(samples), disable=not verbose)
        for sample in samples:
            samples.set_postfix({'RAM': ram(), 'VRAM': vram()})
            sample_and_replace_weights(model, inv_factors, estimator)
            predictions, labels = eval_nn(model, dataset, device)
            mean_predictions += predictions
            model.load_state_dict(mean_state)

            if stats:
                running_mean = mean_predictions / (sample + 1)
                stats_list["acc"].append(accuracy(running_mean, labels))
                stats_list["ece"].append(100 * expected_calibration_error(running_mean, labels)[0])
                stats_list["nll"].append(negative_log_likelihood(predictions, labels))
                stats_list["ent"].append(predictive_entropy(running_mean, mean=True))
        mean_predictions /= len(samples)

        if verbose:
            print(f"Accuracy: {accuracy(mean_predictions, labels):.2f}% | ECE: {100 * expected_calibration_error(mean_predictions, labels)[0]:.2f}%")

        return mean_predictions, labels, stats_list


def eval_nn_and_bnn(model,
                    dataset,
                    inv_factors,
                    estimator,
                    samples,
                    stats,
                    device=torch.device('cuda'),
                    verbose=True):

    predictions, labels = eval_nn(model, dataset, device, verbose)
    try:
        dataset.dataset.set_use_cache(True)
    except AttributeError:
        pass
    bnn_predictions, _, bnn_stats = eval_bnn(model, dataset, inv_factors, estimator, samples, stats, device, verbose)

    return predictions, bnn_predictions, labels, bnn_stats


def test(args, model, fig_path=""):
    print("Loading data")
    if args.data == 'cifar10':
        test_loader = datasets.cifar10(args.torch_data, splits='test')
    elif args.data == 'gtsrb':
        test_loader = datasets.gtsrb(args.data_dir, batch_size=args.batch_size, splits='test')
    if args.data == 'mnist':
        test_loader = datasets.mnist(args.torch_data, splits='test')
    elif args.data == 'tiny':
        test_loader = datasets.imagenet(args.data_dir, img_size=64, batch_size=args.batch_size, splits='test',
                                        tiny=True)
    elif args.data == 'imagenet':
        img_size = 224
        if args.model in ['googlenet', 'inception_v3']:
            img_size = 299
        test_loader = datasets.imagenet(args.data_dir, img_size, args.batch_size, workers=args.workers, splits='test')

    predictions, labels = eval_nn(model, test_loader, args.device, args.verbose)

    print("Plotting results")
    plot.reliability_diagram(predictions, labels, path=fig_path + "_reliability.pdf")


def out_of_domain(args, model, inv_factors, results_path="", fig_path=""):
    """Evaluates the model on in- and out-of-domain data.

    Each dataset has its own out-of-domain dataset which is loaded automatically alongside the in-domain dataset
    specified in `args.data`. For each image (batch) in the in- and out-of-domain data a forward pass through the
    provided `model` is performed and the predictions are stored under `results_path`. This is repeated for the Bayesian
    variant of the model (Laplace approximation).

    Parameters
    ----------
    args : Todo: Check type
        The arguments provided to the script on execution.
    model : torch.nn.Module Todo: Verify
        A `torchvision` or custom neural network (a `torch.nn.Module` or `torch.nn.Sequential` instance)
    inv_factors : list
        A list KFAC factors, Eigenvectors of KFAC factors or diagonal terms. Todo: INF
    results_path : string, optional
        The path where results (in- and out-of-domain predictions) should be stored. Results are not stored if
        argument `args.no_results` is provided.
    fig_path : string, optional
        The path where figures should be stored. Figures are only generated if argument `args.plot` is provided.
    """
    print("Loading data")
    if args.data == 'cifar10':
        in_data = datasets.cifar10(args.torch_data, splits='test')
        out_data = datasets.svhn(args.torch_data, splits='test')
    elif args.data == 'mnist':
        in_data = datasets.mnist(args.torch_data, splits='test')
        out_data = datasets.kmnist(args.torch_data, splits='test')
    elif args.data == 'gtsrb':
        in_data = datasets.gtsrb(args.data_dir, batch_size=args.batch_size, splits='test')
        out_data = datasets.cifar10(args.torch_data, splits='test')
    elif args.data == 'tiny':
        in_data = datasets.imagenet(args.data_dir, img_size=64, batch_size=args.batch_size, splits='test', tiny=True,
                                    use_cache=True)
        out_data = datasets.art(args.data_dir, img_size=64, batch_size=args.batch_size, use_cache=True)
    elif args.data == 'imagenet':
        img_size = 224
        if args.model in ['googlenet', 'inception_v3']:
            img_size = 299
        in_data = datasets.imagenet(args.data_dir, img_size, args.batch_size, workers=args.workers, splits='test',
                                    use_cache=True)
        out_data = datasets.art(args.data_dir, img_size, args.batch_size, workers=args.workers, use_cache=True)

    # Compute NN and BNN predictions on validation set of training data
    predictions, bnn_predictions, labels, stats = eval_nn_and_bnn(model, in_data, inv_factors, args.estimator,
                                                                  args.samples, args.stats, args.device, verbose=True)

    # Compute NN and BNN predictions on out-of-distribution data
    ood_predictions, bnn_ood_predictions, _, _ = eval_nn_and_bnn(model, out_data, inv_factors, args.estimator,
                                                                 args.samples, False, args.device, verbose=True)

    if not args.no_results:
        print("Saving results")
        np.savez_compressed(results_path,
                            stats=stats,
                            labels=labels,
                            predictions=predictions,
                            bnn_predictions=bnn_predictions,
                            ood_predictions=ood_predictions,
                            bnn_ood_predictions=bnn_ood_predictions)

    if args.plot:
        print("Plotting results")
        fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)
        plot.inv_ecdf_vs_pred_entropy(predictions, color='dodgerblue', linestyle='--', axis=ax)
        plot.inv_ecdf_vs_pred_entropy(ood_predictions, color='crimson', linestyle='--', axis=ax)
        plot.inv_ecdf_vs_pred_entropy(bnn_predictions, color='dodgerblue', axis=ax)
        plot.inv_ecdf_vs_pred_entropy(bnn_ood_predictions, color='crimson', axis=ax)
        ax.legend([f"NN {args.data.upper()} | Acc.: {accuracy(predictions, labels):.2f}%",
                   f"NN OOD",
                   f"BNN {args.data.upper()} | Acc.: {accuracy(bnn_predictions, labels):.2f}%",
                   f"BNN OOD"], fontsize=16, frameon=False)
        plt.savefig(fig_path + "_ecdf.pdf", forma='pdf', dpi=1200)

        plot.reliability_diagram(predictions, labels, path=fig_path + "_reliability.pdf")
        plot.reliability_diagram(bnn_predictions, labels, path=fig_path + "_bnn_reliability.pdf")

        plot.entropy_hist(predictions, ood_predictions, path=fig_path + "_entropy.pdf")
        plot.entropy_hist(bnn_predictions, bnn_ood_predictions, path=fig_path + "_bnn_entropy.pdf")


def adversarial_attack(args, model, inv_factors, results_path, fig_path):
    print("Loading data")
    if args.data == 'cifar10':
        test_loader = datasets.cifar10(args.torch_data, splits='test')
    elif args.data == 'gtsrb':
        test_loader = datasets.gtsrb(args.data_dir, batch_size=args.batch_size, splits='test')
    if args.data == 'mnist':
        test_loader = datasets.mnist(args.torch_data, splits='test')
    elif args.data == 'tiny':
        test_loader = datasets.imagenet(args.data_dir, img_size=64, batch_size=args.batch_size, splits='test',
                                        tiny=True)
    elif args.data == 'imagenet':
        img_size = 224
        if args.model in ['googlenet', 'inception_v3']:
            img_size = 299
        test_loader = datasets.imagenet(args.data_dir, img_size, args.batch_size, workers=args.workers, splits='test')

    if args.epsilon > 0:
        print(eval_fgsm(model, test_loader, args.epsilon, args.device)[-1])
    else:
        stats_dict = {"eps": [], "acc": [], "ece1": [], "ece2": [], "nll": [], "ent": []}
        bnn_stats_dict = {"eps": [], "acc": [], "ece1": [], "ece2": [], "nll": [], "ent": []}
        steps = np.concatenate([np.linspace(0, 0.2, 11), np.linspace(0.3, 1, 8)])
        for step in steps:
            stats = eval_fgsm(model, test_loader, step, args.device, verbose=False)[-1]
            bnn_stats = eval_fgsm_bnn(model, test_loader, inv_factors, args.estimator, args.samples, step,
                                      device=args.device)[-1]
            for (k1, v1), (k2, v2) in zip(stats.items(), bnn_stats.items()):
                stats_dict[k1].append(v1)
                bnn_stats_dict[k2].append(v2)
            np.savez(results_path + "_fgsm.npz", stats=stats_dict, bnn_stats=bnn_stats_dict)
        print(tabulate.tabulate(stats_dict, headers="keys"))
        print(tabulate.tabulate(bnn_stats_dict, headers="keys"))

        plot.adversarial_results(steps, stats_dict, bnn_stats_dict, fig_path)


def main():
    args = setup()

    print("Preparing directories")
    os.makedirs(os.path.join(args.results_dir, args.model, "data", args.estimator), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, args.model, "figures", args.estimator), exist_ok=True)
    filename = f"{args.prefix}{args.model}_{args.data}{args.suffix}"
    results_path = os.path.join(args.results_dir, args.model, "data", args.estimator, filename)
    param_path = os.path.join(args.results_dir, args.model, "data", args.estimator, args.optimizer, filename)
    fig_path = os.path.join(args.results_dir, args.model, "figures", args.estimator, filename)

    print("Loading model")
    if args.model == 'lenet5':
        model = lenet5(pretrained=args.data, device=args.device)
    elif args.model == 'resnet18' and args.data != 'imagenet':
        model = resnet18(pretrained=os.path.join(args.root_dir, 'weights', f"{args.model}_{args.data}.pth"),
                         num_classes=43 if args.data == 'gtsrb' else 10, device=args.device)
    else:
        model_class = getattr(torchvision.models, args.model)
        if args.model in ['googlenet', 'inception_v3']:
            model = model_class(pretrained=True, aux_logits=False)
        else:
            model = model_class(pretrained=True)
    model.to(args.device).eval()
    if args.parallel:
        model = torch.nn.parallel.DataParallel(model)

    if args.ood or args.fgsm:
        print("Loading factors")
        factors_path = os.path.join(args.root_dir, "factors", f"{args.model}_{args.data}_{args.estimator}")
        if args.estimator in ['diag', 'kfac']:
            factors = torch.load(factors_path + '.pth')
        elif args.estimator == 'efb':
            kfac_factors = torch.load(factors_path.replace("efb", "kfac") + '.pth')
            lambdas = torch.load(factors_path + '.pth')

            factors = list()
            eigvecs = get_eigenvectors(kfac_factors)

            for eigvec, lambda_ in zip(eigvecs, lambdas):
                factors.append([eigvec[0], eigvec[1], lambda_])
        elif args.estimator == 'inf':
            try:
                factors = torch.load(f"{factors_path}{args.rank}.pth")
            except FileNotFoundError:
                factors = np.load(factors_path + f"{args.rank}.npz", allow_pickle=True)['sif_list']  # Todo: Remove

        print("Inverting factors")
        if args.norm == -1 or args.scale == -1:
            norm, scale = np.load(param_path + "_best_params.npy")
        else:
            norm, scale = args.norm, args.scale
        scale = args.pre_scale * scale
        inv_factors = invert_factors(factors, norm, scale, args.estimator)

    if args.fgsm:
        adversarial_attack(args, model, inv_factors, results_path, fig_path)
    elif args.ood:
        out_of_domain(args, model, inv_factors, results_path, fig_path)
    else:
        fig_path = os.path.join(args.results_dir, args.model, "figures", filename)
        test(args, model, fig_path)


if __name__ == "__main__":
    main()
