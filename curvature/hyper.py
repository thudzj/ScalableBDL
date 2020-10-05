import os
import warnings

import numpy as np
import skopt
import torch
import torchvision
import tabulate

import datasets
from lenet5 import lenet5
from resnet import resnet18
from evaluate import eval_bnn
from sampling import invert_factors
from utils import (accuracy, setup, expected_calibration_error, predictive_entropy, negative_log_likelihood,
                   get_eigenvectors)
from visualize import hyperparameters


def grid(func, dimensions):
    cost = list()
    norms, scales = dimensions
    for norm in norms:
        for scale in scales:
            cost.append(func(norm, scale))
    return cost


def main():
    args = setup()

    print("Preparing directories")
    filename = f"{args.prefix}{args.model}_{args.data}_{args.estimator}{args.suffix}"
    factors_path = os.path.join(args.root_dir, "factors", filename)
    weights_path = os.path.join(args.root_dir, "weights", f"{args.model}_{args.data}.pth")
    if args.exp_id == -1:
        if not args.no_results:
            os.makedirs(os.path.join(args.results_dir, args.model, "data", args.estimator, args.optimizer), exist_ok=True)
        if args.plot:
            os.makedirs(os.path.join(args.results_dir, args.model, "figures", args.estimator, args.optimizer), exist_ok=True)
        results_path = os.path.join(args.results_dir, args.model, "data", args.estimator, args.optimizer, filename)
    else:
        if not args.no_results:
            os.makedirs(os.path.join(args.results_dir, args.model, "data", args.estimator, args.optimizer, args.exp_id), exist_ok=True)
        if args.plot:
            os.makedirs(os.path.join(args.results_dir, args.model, "figures", args.estimator, args.optimizer, args.exp_id), exist_ok=True)
        results_path = os.path.join(args.results_dir, args.model, "data", args.estimator, args.optimizer, args.exp_id, filename)

    print("Loading model")
    if args.model == 'lenet5':
        model = lenet5(pretrained=args.data, device=args.device)
    elif args.model == 'resnet18' and args.data != 'imagenet':
        model = resnet18(pretrained=weights_path, num_classes=43 if args.data == 'gtsrb' else 10, device=args.device)
    else:
        model_class = getattr(torchvision.models, args.model)
        if args.model in ['googlenet', 'inception_v3']:
            model = model_class(pretrained=True, aux_logits=False)
        else:
            model = model_class(pretrained=True)
    model.to(args.device).eval()
    if args.parallel:
        model = torch.nn.parallel.DataParallel(model)

    print("Loading data")
    if args.data == 'mnist':
        val_loader = datasets.mnist(args.torch_data, splits='val')
    elif args.data == 'cifar10':
        val_loader = datasets.cifar10(args.torch_data, splits='val')
    elif args.data == 'gtsrb':
        val_loader = datasets.gtsrb(args.data_dir, batch_size=args.batch_size, splits='val')
    elif args.data == 'imagenet':
        img_size = 224
        if args.model in ['googlenet', 'inception_v3']:
            img_size = 299
        val_loader = datasets.imagenet(args.data_dir, img_size, args.batch_size, args.workers, splits='val',
                                       use_cache=True, pre_cache=True)

    print("Loading factors")
    if args.estimator in ["diag", "kfac"]:
        factors = torch.load(factors_path + '.pth')
    elif args.estimator == 'efb':
        kfac_factors = torch.load(factors_path.replace("efb", "kfac") + '.pth')
        lambdas = torch.load(factors_path + '.pth')

        factors = list()
        eigvecs = get_eigenvectors(kfac_factors)

        for eigvec, lambda_ in zip(eigvecs, lambdas):
            factors.append((eigvec[0], eigvec[1], lambda_))
    elif args.estimator == 'inf':
        factors = torch.load(f"{factors_path}{args.rank}.pth")
    torch.backends.cudnn.benchmark = True

    norm_min = -10
    norm_max = 10
    scale_min = -10
    scale_max = 10
    if args.boundaries:
        x0 = list()
        boundaries = [[norm_min, scale_min],
                      [norm_max, scale_max],
                      [norm_min, scale_max],
                      [norm_max, scale_min],
                      [norm_min / 2., scale_min],
                      [norm_max / 2., scale_max],
                      [norm_min, scale_max / 2.],
                      [norm_max, scale_min / 2.],
                      [norm_min / 2., scale_min / 2.],
                      [norm_max / 2., scale_max / 2.],
                      [norm_min / 2., scale_max / 2.],
                      [norm_max / 2., scale_min / 2.]]
        for b in boundaries:
            tmp = list()
            for _ in range(3 if args.layer else 1):
                tmp.extend(b)
            x0.append(tmp)
    else:
        x0 = None
    f_norms = np.array([factor.norm().cpu().numpy() for factor in factors])

    space = list()
    for i in range(3 if args.layer else 1):
        space.append(skopt.space.Real(norm_min, norm_max, name=f"norm{i}", prior='uniform'))
        space.append(skopt.space.Real(scale_min, scale_max, name=f"scale{i}", prior='uniform'))

    stats = {"norms": [], "scales": [], "acc": [], "ece": [], "nll": [], "ent": [], "cost": []}

    @skopt.utils.use_named_args(dimensions=space)
    def objective(**params):
        norms = list()
        scales = list()
        for f in f_norms:
            if args.layer:
                # Closest to max
                if abs(f_norms.max() - f) < abs(f_norms.min() - f) and abs(f_norms.max() - f) < abs(f_norms.mean() - f):
                    norms.append(10 ** params['norm0'])
                    scales.append(10 ** params['scale0'])
                # Closest to min
                elif abs(f_norms.min() - f) < abs(f_norms.max() - f) and abs(f_norms.min() - f) < abs(f_norms.mean() - f):
                    norms.append(10 ** params['norm1'])
                    scales.append(10 ** params['scale1'])
                # Closest to mean
                else:
                    norms.append(10 ** params['norm2'])
                    scales.append(10 ** params['scale2'])
            else:
                norms.append(10 ** params['norm0'])
                scales.append(10 ** params['scale0'])
        if args.layer:
            print(tabulate.tabulate({'Layer': np.arange(len(factors)),
                                     'F-Norm:': f_norms,
                                     'Norms': norms,
                                     'Scales': scales},
                                    headers='keys', numalign='right'))
        else:
            print("Norm:", norms[0], "Scale:", scales[0])
        try:
            inv_factors = invert_factors(factors, norms, args.pre_scale * scales, args.estimator)
        except (RuntimeError, np.linalg.LinAlgError):
            print(f"Error: Singular matrix")
            return 200

        predictions, labels, _ = eval_bnn(model, val_loader, inv_factors, args.estimator, args.samples, stats=False,
                                          device=args.device, verbose=False)

        err = 100 - accuracy(predictions, labels)
        ece = 100 * expected_calibration_error(predictions, labels)[0]
        nll = negative_log_likelihood(predictions, labels)
        ent = predictive_entropy(predictions, mean=True)
        stats["norms"].append(norms)
        stats["scales"].append(scales)
        stats["acc"].append(100 - err)
        stats["ece"].append(ece)
        stats["nll"].append(nll)
        stats["ent"].append(ent)
        stats["cost"].append(err + ece)
        print(f"Err.: {err:.2f}% | ECE: {ece:.2f}% | NLL: {nll:.3f} | Ent.: {ent:.3f}")

        return err + ece

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        if args.optimizer == "gbrt":
            res = skopt.gbrt_minimize(func=objective, dimensions=space, n_calls=args.calls, x0=x0, verbose=True,
                                      n_jobs=args.workers, n_random_starts=0 if x0 else 10, acq_func='EI')

        # EI (neg. expected improvement)
        # LCB (lower confidence bound)
        # PI (neg. prob. of improvement): Usually favours exploitation over exploration
        # gp_hedge (choose probabilistically between all)
        if args.optimizer == "gp":
            res = skopt.gp_minimize(func=objective, dimensions=space, n_calls=args.calls, x0=x0, verbose=True,
                                    n_jobs=args.workers, n_random_starts=0 if x0 else 1, acq_func='gp_hedge')

        # acq_func: EI (neg. expected improvement), LCB (lower confidence bound), PI (neg. prob. of improvement)
        # xi: how much improvement one wants over the previous best values.
        # kappa: Importance of variance of predicted values. High: exploration > exploitation
        # base_estimator: RF (random forest), ET (extra trees)
        elif args.optimizer == "forest":
            res = skopt.forest_minimize(func=objective, dimensions=space, n_calls=args.calls, x0=x0, verbose=True,
                                        n_jobs=args.workers, n_random_starts=0 if x0 else 1, acq_func='EI')

        elif args.optimizer == "random":
            res = skopt.dummy_minimize(func=objective, dimensions=space, n_calls=args.calls, x0=x0, verbose=True)

        elif args.optimizer == "grid":
            space = [np.arange(norm_min, norm_max + 1, 10), np.arange(scale_min, scale_max + 1, 10)]
            res = grid(func=objective, dimensions=space)

        print(f"Minimal cost of {min(stats['cost'])} found at:")
        if args.layer:
            print(tabulate.tabulate({'Layer': np.arange(len(factors)),
                                     'F-Norm:': f_norms,
                                     'Norms': stats['norms'][np.argmin(stats['cost'])],
                                     'Scales': stats['scales'][np.argmin(stats['cost'])]},
                                    headers='keys', numalign='right'))
        else:
            print("Norm:", stats['norms'][np.argmin(stats['cost'])][0],
                  "Scale:", stats['scales'][np.argmin(stats['cost'])][0])

    if not args.no_results:
        print("Saving results")
        del res.specs['args']['func']
        np.save(results_path + f"_best_params{'_layer.npy' if args.layer else '.npy'}",
                [stats['norms'][np.argmin(stats['cost'])],
                stats['scales'][np.argmin(stats['cost'])]])
        np.save(results_path + f"_hyperopt_stats{'_layer.npy' if args.layer else '.npy'}", stats)
        skopt.dump(res, results_path + f"_hyperopt_dump{'_layer.pkl' if args.layer else '.pkl'}")

    if args.plot:
        print("Plotting results")
        hyperparameters(args)


if __name__ == "__main__":
    main()
