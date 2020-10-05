import os
from typing import Union, Any

import torch
import torchvision
import tqdm

import datasets
import lenet5
import resnet
import fisher
from utils import setup, ram, vram


def compute_inf(args):
    print("Loading factors")
    factors_path = os.path.join(args.root_dir, "factors", f"{args.prefix}{args.model}_{args.data}_kfac{args.suffix}")
    factors = torch.load(factors_path + '.pth')

    print("Loading lambdas")
    factors_path = os.path.join(args.root_dir, "factors", f"{args.prefix}{args.model}_{args.data}_efb{args.suffix}")
    lambdas = torch.load(factors_path + '.pth')

    print("Loading diags")
    factors_path = os.path.join(args.root_dir, "factors", f"{args.prefix}{args.model}_{args.data}_diag{args.suffix}")
    diags = torch.load(factors_path + '.pth')

    print("Computing inf")
    inf = fisher.INF(factors, lambdas, diags)
    inf.accumulate(args.rank)

    return inf


def compute_factors(args: Any,
                    model: Union[torch.nn.Module, torch.nn.Sequential],
                    data: iter,
                    factors=None):

    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    est_base = getattr(fisher, args.estimator.upper())
    if args.estimator == 'efb':
        est = est_base(model, factors)
    else:
        est = est_base(model)

    for epoch in range(args.epochs):
        data = tqdm.tqdm(data, desc=f"Epoch [{epoch + 1}/{args.epochs}]", disable=not args.verbose)
        for batch, (images, labels) in enumerate(data):
            data.set_postfix({'RAM': ram(), 'VRAM': vram()})

            logits = model(images.to(args.device, non_blocking=True))
            dist = torch.distributions.Categorical(logits=logits)

            for sample in range(args.samples):
                labels = dist.sample()

                loss = criterion(logits, labels)
                model.zero_grad()
                loss.backward(retain_graph=True)

                est.update(images.size(0))
    return est


def main():
    args = setup()

    print("Preparing directories")
    os.makedirs(os.path.join(args.root_dir, "factors"), exist_ok=True)
    filename = f"{args.prefix}{args.model}_{args.data}_{args.estimator}{args.suffix}"
    factors_path = os.path.join(args.root_dir, "factors", filename)

    print("Loading model")
    if args.model == 'lenet5':
        model = lenet5.lenet5(pretrained=args.data, device=args.device)
    elif args.model == 'resnet18' and args.data != 'imagenet':
        model = resnet.resnet18(pretrained=os.path.join(args.root_dir, 'weights', f"{args.model}_{args.data}.pth"),
                                num_classes=43 if args.data == 'gtsrb' else 10, device=args.device)
    else:
        model_class = getattr(torchvision.models, args.model)
        if args.model in ['googlenet', 'inception_v3']:
            model = model_class(pretrained=True, aux_logits=False)
        else:
            model = model_class(pretrained=True)
    model.to(args.device).train()
    if args.parallel:
        model = torch.nn.parallel.DataParallel(model)

    if args.estimator != 'inf':
        print(f"Loading data")
        if args.data == 'cifar10':
            data = datasets.cifar10(args.torch_data, args.batch_size, args.workers, args.augment,
                                    splits='train')
        elif args.data == 'mnist':
            data = datasets.mnist(args.torch_data, args.batch_size, args.workers, args.augment, splits='train')
        elif args.data == 'gtsrb':
            data = datasets.gtsrb(args.data_dir, batch_size=args.batch_size, workers=args.workers, splits='train')
        elif args.data == 'tiny':
            img_size = 64
            data = datasets.imagenet(args.data_dir, img_size, args.batch_size, splits='train', tiny=True)
        elif args.data == 'imagenet':
            img_size = 224
            if args.model in ['googlenet', 'inception_v3']:
                img_size = 299
            data = datasets.imagenet(args.data_dir, img_size, args.batch_size, workers=args.workers, splits='train')
    torch.backends.cudnn.benchmark = True

    print("Computing factors")
    if args.estimator == 'inf':
        est = compute_inf(args)
    elif args.estimator == 'efb':
        factors = torch.load(factors_path.replace("efb", "kfac") + '.pth')
        est = compute_factors(args, model, data, factors)
    else:
        est = compute_factors(args, model, data)

    print("Saving factors")
    if args.estimator == "inf":
        torch.save(est.state, f"{factors_path}{args.rank}.pth")
    elif args.estimator == "efb":
        torch.save(list(est.state.values()), factors_path + '.pth')
        torch.save(list(est.diags.values()), factors_path.replace("efb", "diag") + '.pth')
    else:
        torch.save(list(est.state.values()), factors_path + '.pth')


if __name__ == "__main__":
    main()
