import os

import numpy as np
import skopt
import torch
import torchvision
from matplotlib import pyplot as plt
import tabulate

import plot
from lenet5 import lenet5
from resnet import resnet18
from utils import setup, get_eigenvalues, accuracy, get_eigenvectors

import matplotlib
matplotlib.use("TkAgg")


def load_data(args, model, data='imagenet', estimator="kfac", ood=False):
    if estimator in ["swag", "swa"] and data == 'imagenet':
        data_path = os.path.join(args.results_dir, model, "data", "swag", f"ood_{estimator}" if ood else estimator)
        data = np.load(data_path + ".npz")
        return data['targets'], None, data['predictions']

    filename = f"{args.prefix}{model}_{data}{args.suffix}"
    data_path = os.path.join(args.results_dir, model, "data", estimator, filename)
    data = np.load(data_path + ".npz")
    if ood:
        return data['labels'], data['ood_predictions'], data['bnn_ood_predictions']
    return data['labels'], data['predictions'], data['bnn_predictions']


def ecdf(args):
    fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)

    labels, predictions, bnn_predictions = load_data(args, args.model, args.data, args.estimator)
    labels, ood_predictions, bnn_ood_predictions = load_data(args, args.model, args.data, args.estimator, ood=True)

    plot.inv_ecdf_vs_pred_entropy(predictions, color='dodgerblue', linestyle='--', axis=ax)
    plot.inv_ecdf_vs_pred_entropy(ood_predictions, color='crimson', linestyle='--', axis=ax)
    plot.inv_ecdf_vs_pred_entropy(bnn_predictions, color='dodgerblue', axis=ax)
    plot.inv_ecdf_vs_pred_entropy(bnn_ood_predictions, color='crimson', axis=ax)
    ax.legend([f"NN {args.data.upper()} | Acc.: {accuracy(predictions, labels):.2f}%",
               f"NN OOD",
               f"BNN {args.data.upper()} | Acc.: {accuracy(bnn_predictions, labels):.2f}%",
               f"BNN OOD"], fontsize=16, frameon=False)
    filename = f"{args.prefix}{args.model}_{args.data}{args.suffix}"
    plt.savefig(args.results_dir + f"/figures/ecdf_{filename}.pdf", format='pdf', dpi=1200)


def entropy_histogram(args):
    if args.estimator in ['swa', 'swag']:
        _, _, predictions = load_data(args, args.model, args.data, args.estimator)
        _, _, ood_predictions = load_data(args, args.model, args.data, args.estimator, ood=True)
    elif args.estimator == 'sgd':
        _, predictions, _ = load_data(args, args.model, args.data, 'kfac')
        _, ood_predictions, _ = load_data(args, args.model, args.data, 'kfac', ood=True)
    else:
        _, _, predictions = load_data(args, args.model, args.data, args.estimator)
        _, _, ood_predictions = load_data(args, args.model, args.data, args.estimator, ood=True)

    fig_path = os.path.join(args.results_dir, args.model, "figures", args.estimator)
    os.makedirs(fig_path, exist_ok=True)

    filename = f"{args.prefix}{args.model}_{args.data}{args.suffix}"
    plot.entropy_hist(predictions, ood_predictions, path=os.path.join(fig_path, f"{filename}_entropy.pdf"))


def calibration(args):
    fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)

    c1 = next(ax._get_lines.prop_cycler)['color']
    c2 = next(ax._get_lines.prop_cycler)['color']
    c3 = next(ax._get_lines.prop_cycler)['color']
    c4 = next(ax._get_lines.prop_cycler)['color']
    c5 = next(ax._get_lines.prop_cycler)['color']
    c6 = next(ax._get_lines.prop_cycler)['color']
    c7 = next(ax._get_lines.prop_cycler)['color']

    try:
        labels, predictions, kfac_predictions = load_data(args, args.model, args.data, "kfac")
        plot.calibration(predictions, labels, color=c1, label=f"SGD", axis=ax)
        plot.calibration(kfac_predictions, labels, color=c2, label="KFAC-Laplace", axis=ax)
    except FileNotFoundError:
        print(f"Data for model {args.model} and estimator SGD/KFAC not available.")
    try:
        diag_labels, _, diag_predictions = load_data(args, args.model, args.data, "diag")
        plot.calibration(diag_predictions, diag_labels, color=c3, label="Diag-Laplace", axis=ax)
    except FileNotFoundError:
        print(f"Data for model {args.model} and estimator DIAG not available.")
    try:
        efb_labels, _, efb_predictions = load_data(args, args.model, args.data, "efb")
        plot.calibration(efb_predictions, efb_labels, color=c4, label="EFB-Laplace", axis=ax)
    except FileNotFoundError:
        print(f"Data for model {args.model} and estimator EFB not available.")
    try:
        inf_labels, _, inf_predictions = load_data(args, args.model, args.data, "inf")
        plot.calibration(inf_predictions, inf_labels, color=c5, label="INF-Laplace", axis=ax)
    except FileNotFoundError:
        print(f"Data for model {args.model} and estimator INF not available.")
    try:
        swag_labels, _, swag_predictions = load_data(args, args.model, args.data, "swag")
        swa_labels, _, swa_predictions = load_data(args, args.model, args.data, "swa")
        plot.calibration(swa_predictions, swa_labels, color=c6, label="SWA", axis=ax)
        plot.calibration(swag_predictions, swag_labels, color=c7, label="SWAG", axis=ax)
    except FileNotFoundError:
        print(f"Data for model {args.model} and estimator SWAG/SWA not available.")

    filename = f"{args.prefix}{args.model}_{args.data}{args.suffix}"
    plt.savefig(args.results_dir + f"/figures/calibration_{filename}.pdf", format='pdf', dpi=1200)


def calibration_overview(args):
    fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)

    all_nets = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                'densenet121', 'densenet161', 'densenet169', 'densenet201',
                'googlenet_x', 'inception_v3', 'wide_resnet50_2', 'vgg16', 'alexnet_x']

    nets = all_nets
    det_list = tuple(all_nets) if args.estimator != 'none' else all_nets
    bnn_list = all_nets if args.estimator != 'none' else []
    for name in nets:
        try:
            labels, predictions, bnn_predictions = load_data(args, name, args.data,
                                                             args.estimator if args.estimator != 'none' else 'kfac')
            color = next(ax._get_lines.prop_cycler)['color']

            if name in det_list and isinstance(det_list, list):
                plot.calibration(predictions, labels, color=color, label=name, axis=ax)
            elif name in bnn_list and isinstance(bnn_list, list):
                plot.calibration(bnn_predictions, labels, color=color, label=name, axis=ax)

            if name in det_list and isinstance(det_list, tuple):
                plot.calibration(predictions, labels, axis=ax, alpha=0.3, color=color, linestyle='--')
            elif name in bnn_list and isinstance(bnn_list, tuple):
                plot.calibration(bnn_predictions, labels, axis=ax, alpha=0.3, color=color, linestyle='--')
        except FileNotFoundError:
            print(f"Data for model {name} not available.")
    plt.savefig(
        args.results_dir + f"/figures/calibration_overview_{args.estimator if args.estimator != 'none' else ''}.pdf",
        format='pdf', dpi=1200)


def out_of_domain(args):
    fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)

    c1 = next(ax._get_lines.prop_cycler)['color']
    c2 = next(ax._get_lines.prop_cycler)['color']
    c3 = next(ax._get_lines.prop_cycler)['color']
    c4 = next(ax._get_lines.prop_cycler)['color']
    c5 = next(ax._get_lines.prop_cycler)['color']
    c6 = next(ax._get_lines.prop_cycler)['color']
    c7 = next(ax._get_lines.prop_cycler)['color']

    try:
        labels, predictions, kfac_predictions = load_data(args, args.model, args.data, "kfac")
        ood_labels, ood_predictions, ood_kfac_predictions = load_data(args, args.model, args.data, "kfac", ood=True)
        plot.inv_ecdf_vs_pred_entropy(ood_predictions, "SGD", c1, axis=ax)
        plot.inv_ecdf_vs_pred_entropy(ood_kfac_predictions, "KFAC-Laplace", c2, axis=ax)
        plot.inv_ecdf_vs_pred_entropy(predictions, color=c1, linestyle='--', axis=ax)
        plot.inv_ecdf_vs_pred_entropy(kfac_predictions, color=c2, linestyle='--', axis=ax)
    except FileNotFoundError:
        print(f"Data for model {args.model} and estimator SGD/KFAC not available.")
    try:
        diag_labels, _, diag_predictions = load_data(args, args.model, args.data, "diag")
        ood_diag_labels, _, ood_diag_predictions = load_data(args, args.model, args.data, "diag", ood=True)
        plot.inv_ecdf_vs_pred_entropy(ood_diag_predictions, "Diag-Laplace", c3, axis=ax)
        plot.inv_ecdf_vs_pred_entropy(diag_predictions, color=c3, linestyle='--', axis=ax)
    except FileNotFoundError:
        print(f"Data for model {args.model} and estimator DIAG not available.")
    try:
        efb_labels, _, efb_predictions = load_data(args, args.model, args.data, "efb")
        ood_efb_labels, _, ood_efb_predictions = load_data(args, args.model, args.data, "efb", ood=True)
        plot.inv_ecdf_vs_pred_entropy(ood_efb_predictions, "EFB-Laplace", c4, axis=ax)
        plot.inv_ecdf_vs_pred_entropy(efb_predictions, color=c4, linestyle='--', axis=ax)
    except FileNotFoundError:
        print(f"Data for model {args.model} and estimator EFB not available.")
    try:
        inf_labels, _, inf_predictions = load_data(args, args.model, args.data, "inf")
        ood_inf_labels, _, ood_inf_predictions = load_data(args, args.model, args.data, "inf", ood=True)
        plot.inv_ecdf_vs_pred_entropy(ood_inf_predictions, "INF-Laplace", c5, axis=ax)
        plot.inv_ecdf_vs_pred_entropy(inf_predictions, color=c5, linestyle='--', axis=ax)
    except FileNotFoundError:
        print(f"Data for model {args.model} and estimator INF not available.")
    try:
        swag_labels, _, swag_predictions = load_data(args, args.model, args.data, "swag")
        swa_labels, _, swa_predictions = load_data(args, args.model, args.data, "swa")
        ood_swag_labels, _, ood_swag_predictions = load_data(args, args.model, args.data, "swag", ood=True)
        ood_swa_labels, _, ood_swa_predictions = load_data(args, args.model, args.data, "swa", ood=True)
        plot.inv_ecdf_vs_pred_entropy(ood_swa_predictions, "SWA", c6, axis=ax)
        plot.inv_ecdf_vs_pred_entropy(ood_swag_predictions, "SWAG", c7, axis=ax)
        plot.inv_ecdf_vs_pred_entropy(swa_predictions, color=c6, linestyle='--', axis=ax)
        plot.inv_ecdf_vs_pred_entropy(swag_predictions, color=c7, linestyle='--', axis=ax)
    except FileNotFoundError:
        print(f"Data for model {args.model} and estimator SWA/SWAG not available.")

    solid, = ax.plot([], [], c="black", ls='-')
    dashed, = ax.plot([], [], c="black", ls='--')

    lines = plt.legend(fontsize=16, loc="upper right", frameon=False)
    plt.legend([solid, dashed], ["out-of-domain", "in-domain"], fontsize=14, loc="lower left", frameon=False)
    ax.add_artist(lines)

    plt.savefig(args.results_dir + f"/figures/out_of_domain_{args.model}.pdf", format='pdf', dpi=1200)


def out_of_domain_overview(args):
    fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)

    nets = ['resnet18', 'resnet50', 'resnet152', 'densenet121', 'densenet161']

    for name in nets:
        color = next(ax._get_lines.prop_cycler)['color']
        try:
            if args.estimator in ['swa', 'swag']:
                _, _, predictions = load_data(args, name, args.data, args.estimator)
                _, _, ood_predictions = load_data(args, name, args.data, args.estimator, ood=True)
            elif args.estimator == 'sgd':
                _, predictions, _ = load_data(args, name, args.data, 'kfac')
                _, ood_predictions, _ = load_data(args, name, args.data, 'kfac', ood=True)
            else:
                _, _, predictions = load_data(args, name, args.data, args.estimator)
                _, _, ood_predictions = load_data(args, name, args.data, args.estimator, ood=True)
            plot.inv_ecdf_vs_pred_entropy(ood_predictions, name.capitalize(), color=color, axis=ax)
            plot.inv_ecdf_vs_pred_entropy(predictions, color=color, linestyle='--', axis=ax)
        except FileNotFoundError:
            print(f"Data for model {name} not available.")

    solid, = ax.plot([], [], c="black", ls='-')
    dashed, = ax.plot([], [], c="black", ls='--')

    lines = plt.legend(fontsize=16, loc="upper right", frameon=False)
    plt.legend([solid, dashed], ["out-of-domain", "in-domain"], fontsize=14, loc="lower left", frameon=False)
    ax.add_artist(lines)

    plt.savefig(args.results_dir + f"/figures/out_of_domain_{args.estimator}.pdf", format='pdf', dpi=1200)


def eigenvalues(args):
    print("Loading factors")
    factors_path = os.path.join(args.root_dir, "factors",
                                f"{args.prefix}{args.model}_{args.data}_{args.estimator}{args.suffix}.pth")
    factors = torch.load(factors_path, map_location='cpu')

    filename = f"{args.prefix}{args.model}_{args.data}{args.suffix}"
    fig_path = os.path.join(args.results_dir, args.model, "figures", args.estimator, filename)

    print("Computing eigenvalues")
    eigvals = get_eigenvalues(factors, verbose=True)
    plot.eigenvalue_histogram(eigvals.numpy(), show_stats=args.verbose, path=fig_path + "_eigenvalue_histogram.pdf")


def print_hyperparameters(args):
    hyper_list = list()
    for name in ['resnet18', 'resnet50', 'resnet152',
                 'densenet121', 'densenet161']:
        tmp = [name.capitalize()]
        for est in ['diag', 'kfac', 'efb', 'inf']:
            filename = f"{args.prefix}{name}_{args.data}{args.suffix}"
            results_path = os.path.join(args.results_dir, name, "data", est, filename)
            try:
                norm, scale = np.load(results_path + "_best_params.npy")
                tmp.append(round(norm))
                tmp.append(round(scale))
            except FileNotFoundError:
                pass
        hyper_list.append(tmp)
    print(tabulate.tabulate(hyper_list,
                            headers=['Model', 'DIAG Norm', 'DIAG Scale', 'KFAC Norm', 'KFAC Scale',
                                     'EFB Norm', 'EFB Scale', 'INF Norm', 'INF Scale'], tablefmt='rst',
                            numalign='right', floatfmt=".0f"))


def hyperparameters(args, plot_3d=False, plot_time=True):
    filename = f"{args.prefix}{args.model}_{args.data}_{args.estimator}{args.suffix}"
    if not plot_time:
        results_path = os.path.join(args.results_dir, args.model, "data", args.estimator, args.optimizer, filename)
        try:
            data = np.load(results_path + "_hyperopt_stats.npy", allow_pickle=True).item()
            metrics = ['score', 'acc', 'ent', 'nll', 'ece']
        except FileNotFoundError:
            data = dict()
            metrics = ['score']
        res = skopt.load(results_path + "_hyperopt_dump.pkl")
        data['norm'] = np.array(res.x_iters)[:, 0]
        data['scale'] = np.array(res.x_iters)[:, 1]
        data['score'] = np.array(res.func_vals)

    if plot_3d:
        plot.hyper_results_3d(data)
    elif plot_time:
        fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)
        c1 = next(ax._get_lines.prop_cycler)['color']
        c2 = next(ax._get_lines.prop_cycler)['color']
        gp_data = list()
        random_data = list()
        for i in range(10):
            random_path = os.path.join(args.results_dir, args.model, "data", args.estimator, 'random', str(i), filename)
            gp_path = os.path.join(args.results_dir, args.model, "data", args.estimator, 'gp', str(i), filename)
            try:
                data = np.load(random_path + f"_hyperopt_stats{'_layer.npy' if args.layer else '.npy'}", allow_pickle=True).item()
                #ax.scatter(np.arange(len(random_data['cost'])), random_data['cost'], c=c1, marker='x')
                random_data.append([np.nanmin(data['cost'][:i + 1]) for i in range(len(data['cost']))])
                #ax.plot(random_data, c=c1, lw=0.5, ls='--')
            except FileNotFoundError:
                print(f"Data for experiment {i} (random) not available.")
            try:
                data = np.load(gp_path + f"_hyperopt_stats{'_layer.npy' if args.layer else '.npy'}", allow_pickle=True).item()
                #ax.scatter(np.arange(len(gp_data['cost'])), gp_data['cost'], c=c2, marker='+')
                gp_data.append([np.nanmin(data['cost'][:i + 1]) for i in range(len(data['cost']))])
                #ax.plot(gp_data, c=c2, lw=0.5, ls='--')
            except FileNotFoundError:
                print(f"Data for experiment {i} (gp) not available.")

        random_data = np.array(random_data)
        random_mean = np.nanmean(random_data, axis=0)
        random_std = np.nanstd(random_data, axis=0)
        ax.plot(random_mean, c=c1, lw=2, label='Random Search')
        plt.fill_between(np.arange(random_mean.shape[0]), random_mean - random_std, random_mean + random_std, color=c1, alpha=0.2)

        gp_data = np.array(gp_data)
        gp_mean = np.nanmean(gp_data, axis=0)
        gp_std = np.nanstd(gp_data, axis=0)
        ax.plot(gp_mean, c=c2, lw=2, label='Bayesian Optimization')
        plt.fill_between(np.arange(gp_mean.shape[0]), gp_mean - gp_std, gp_mean + gp_std, color=c2, alpha=0.2)
        plt.legend(fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Error + ECE', fontsize=14)
        plt.show()
    else:
        for metric in metrics:
            data[metric] = np.array(data[metric])
            fig_path = results_path.replace("data", "figures") + f"_hyper_results_{metric}.pdf"
            plot.hyper_results(data, metric, annotate=True, path=fig_path)


def adversarial_attack(args):
    for metric in ['ent', 'acc']:
        fig, ax = plt.subplots(figsize=(9, 9), tight_layout=True)
        filename = f"{args.prefix}{args.model}_{args.data}_fgsm{args.suffix}.npz"

        for estimator in ['kfac', 'diag', 'efb', 'inf']:
            results_path = os.path.join(args.results_dir, args.model, "data", estimator, filename)
            try:
                data = np.load(results_path, allow_pickle=True)
                stats, bnn_stats = data['stats'].item(), data['bnn_stats'].item()
                epsilons = stats['eps']
                color = next(ax._get_lines.prop_cycler)['color']

                if estimator == 'kfac':
                    plt.plot(epsilons, stats[metric], color=color, linewidth=3, label='SGD')
                    color = next(ax._get_lines.prop_cycler)['color']
                label = estimator.upper()
                plt.plot(epsilons, bnn_stats[metric], color=color, linewidth=3, label=f"{label}-Laplace")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(linestyle='dashed')
                ax.tick_params(direction='out', labelsize=14, right=False, top=False)
                ax.set_ylabel('Predictive Entropy' if metric == 'ent' else 'Accuracy', fontsize=16)
                ax.set_xlabel('Step size', fontsize=16)
                ax.set_xlim(0, 1)
                ax.legend(fontsize=16)

            except FileNotFoundError:
                print(f"Data for model {args.model} and estimator {estimator.upper()} not available.")
        plt.savefig(args.results_dir + f"/figures/adv_{metric}_{args.model}_{args.data}.pdf", format='pdf', dpi=1200)


def summary(args):
    if args.model == 'lenet5':
        model = lenet5(pretrained=args.data, device=args.device)
    elif args.model == 'resnet18' and args.data != 'imagenet':
        model = resnet18(pretrained=os.path.join(args.root_dir, 'weights', f"{args.model}_{args.data}.pth"),
                         device=args.device)
    else:
        model_class = getattr(torchvision.models, args.model)
        if args.model in ['googlenet', 'inception_v3']:
            model = model_class(pretrained=True, aux_logits=False)
        else:
            model = model_class(pretrained=True)

    module_classes = list()
    for module in model.modules():
        module_class = module.__class__.__name__
        if module_class in ['Linear', 'Conv2d']:
            module_classes.append(module_class)

    diag_list = list()
    kfac_list = list()
    efb_list = list()
    inf_list = list()
    for est in ['diag', 'kfac', 'efb', 'inf']:
        if est == 'diag':
            factors_list = diag_list
        elif est == 'kfac':
            factors_list = kfac_list
        elif est == 'efb':
            factors_list = efb_list
        else:
            factors_list = inf_list

        factors_path = os.path.join(args.root_dir, "factors", f"{args.model}_{args.data}_{est}")
        if est in ["diag", "kfac"]:
            factors = torch.load(factors_path + '.pth', map_location='cpu')
        elif est == 'efb':
            kfac_factors = torch.load(factors_path.replace("efb", "kfac") + '.pth')
            lambdas = torch.load(factors_path + '.pth', map_location='cpu')

            factors = list()
            eigvecs = get_eigenvectors(kfac_factors)

            for eigvec, lambda_ in zip(eigvecs, lambdas):
                factors.append((eigvec[0], eigvec[1], lambda_))
        elif est == 'inf':
            try:
                factors = torch.load(f"{factors_path}{args.rank}.pth", map_location='cpu')
            except FileNotFoundError:
                factors = np.load(factors_path + f"{args.rank}.npz", allow_pickle=True)['sif_list']  # Todo: Remove

        numel_sum = 0
        for index, (cls, factor) in enumerate(zip(module_classes, factors)):
            numel = np.sum([f.numel() for f in factor]).astype(int)
            if est == 'diag':
                factors_list.append([f"{cls} {index}", numel, (numel * 32) / (8 * 1024 ** 2)])
            else:
                factors_list.append([numel, (numel * 32) / (8 * 1024 ** 2)])
            numel_sum += numel
        if est == 'diag':
            factors_list.append(["Total", numel_sum, (numel_sum * 32) / (8 * 1024 ** 2)])
        else:
            factors_list.append([numel_sum, (numel_sum * 32) / (8 * 1024 ** 2)])

    factors_list = np.concatenate([diag_list, kfac_list, efb_list, inf_list], axis=1)

    print(tabulate.tabulate(factors_list, headers=['Layer #', '#Parameters', 'Size (MB)', '#Parameters', 'Size (MB)', '#Parameters', 'Size (MB)', '#Parameters', 'Size (MB)'], floatfmt=".1f",
                            numalign='right', tablefmt='latex'))


def main():
    args = setup()
    if args.calibration:
        calibration(args)
    elif args.networks:
        out_of_domain_overview(args)
        calibration_overview(args)
    elif args.ood:
        out_of_domain(args)
    elif args.eigvals:
        eigenvalues(args)
    elif args.hyper:
        hyperparameters(args)
    elif args.fgsm:
        adversarial_attack(args)
    elif args.ecdf:
        ecdf(args)
    elif args.entropy:
        entropy_histogram(args)
    elif args.summary:
        summary(args)
    else:
        print_hyperparameters(args)


if __name__ == "__main__":
    main()
