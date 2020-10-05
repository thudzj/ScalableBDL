from __future__ import division
import os, sys, shutil, time, random, math, copy

import argparse
import warnings
import numpy as np

import torch
import torch.backends.cudnn as cudnn

import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import torchvision.datasets as dset
import torchvision.transforms as transforms

import curvature

import models
from mean_field import *
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, _ECELoss, plot_mi, plot_ens

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Training script for CIFAR', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', default='./data', type=str, help='Path to dataset')
parser.add_argument('--dataset', metavar='DSET', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between CIFAR/ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='wrn', help='model architecture: ' + ' | '.join(model_names) + ' (default: wrn)')
parser.add_argument('--depth', type=int, metavar='N', default=28)
parser.add_argument('--wide', type=int, metavar='N', default=10)

# Optimization
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')

#Regularization
parser.add_argument('--dropout_rate', type=float, default=0.)

# Checkpoints
parser.add_argument('--save_path', type=str, default='./laplace', help='Folder to save checkpoints and log.')

# Acceleration
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')

# Random seed
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--job-id', type=str, default='')

# Bayesian
parser.add_argument('--kfac_norm', type=float, default=10)
parser.add_argument('--kfac_scale', type=float, default=50000)

# attack settings
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--epsilon_scale', default=1., type=float)
parser.add_argument('--num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003, type=float,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')

# Dist
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-port', default='1234', type=str,
                    help='port used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc = 0

def main():
    args = parser.parse_args()
    if not os.path.isdir(args.data_path): os.makedirs(args.data_path)
    job_id = args.job_id
    args.save_path = args.save_path + job_id
    if not os.path.isdir(args.save_path): os.makedirs(args.save_path)
    args.num_classes = 10 if args.dataset == 'cifar10' else 100

    args.use_cuda = torch.cuda.is_available()
    if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.use_cuda: torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    else:
        args.multiprocessing_distributed = True

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc
    args.gpu = gpu
    assert args.gpu is not None
    print("Use GPU: {} for training".format(args.gpu))

    log = open(os.path.join(args.save_path, 'log{}{}.txt'.format('_seed'+ str(args.manualSeed), '')), 'w')
    log = (log, args.gpu)

    args.bayes = None
    net = models.__dict__[args.arch](args, args.depth, args.wide, args.num_classes)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch  version : {}".format(torch.__version__), log)
    print_log("CuDNN  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Number of parameters: {}".format(sum([p.numel() for p in net.parameters()])), log)
    print_log(str(args), log)

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url+":"+args.dist_port,
                                world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.gpu)
        net.cuda(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
    else:
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)

    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    kfac = curvature.DIAG(net)

    net_dict = net.state_dict()
    net_dict.update({k.replace("module.", ""): v for k,v in torch.load('./ckpts/wrn_28_10_decay0.0002.pth.tar', map_location='cuda:{}'.format(args.gpu))['state_dict'].items()})
    net.load_state_dict(net_dict)

    cudnn.benchmark = True

    train_loader, test_loader, test_loader1, fake_loader, fake_loader2 = load_dataset_ft(args)

    train_acc, train_los = train(train_loader, net, criterion, kfac, args, log)
    factors = list(kfac.state.values())
    inv_factors = curvature.invert_factors(factors, norm=args.kfac_norm, scale=args.kfac_scale, estimator='diag')
    evaluate(test_loader, test_loader1, fake_loader, fake_loader2, net, criterion, inv_factors, args, log)

    log[0].close()

def train(train_loader, model, criterion, kfac, args, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # print(i)
        # if i == 10:
        #     break
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(input)
        dist = torch.distributions.Categorical(logits=output)
        sampled_labels = dist.sample()
        loss = criterion(output, sampled_labels)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.detach().item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        model.zero_grad()
        loss.backward()
        kfac.update(batch_size=input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i == len(train_loader) - 1:
            print_log('Time {batch_time.avg:.3f}   '
                        'Data {data_time.avg:.3f}   '
                        'Loss {loss.avg:.4f}   '
                        'Prec@1 {top1.avg:.3f}   '
                        'Prec@5 {top5.avg:.3f}   '.format(
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1, top5=top5) + time_string(), log)
    return top1.avg, losses.avg

def evaluate(test_loader, test_loader1, fake_loader, fake_loader2, net, criterion, inv_factors, args, log):

    rets = ens_validate(test_loader, net, criterion, inv_factors, args, log, 100)
    print_log('TOP1 average: {:.4f}, ensemble: {:.4f}'.format(rets[:,2].mean(), rets[-1][-3]), log)
    print_log('TOP5 average: {:.4f}, ensemble: {:.4f}'.format(rets[:,3].mean(), rets[-1][-2]), log)
    print_log('LOS  average: {:.4f}, ensemble: {:.4f}'.format(rets[:,1].mean(), rets[-1][-4]), log)
    print_log('ECE  ensemble: {:.4f}'.format(rets[-1][-1]), log)

    ens_validate(fake_loader, net, criterion, inv_factors, args, log, 20, suffix='_fake')
    if args.gpu == 0: print_log('NAT vs. Fake (SNGAN): AP {}'.format(plot_mi(args.save_path, 'fake')), log)

    ens_validate(fake_loader2, net, criterion, inv_factors, args, log, 20, suffix='_fake2')
    if args.gpu == 0: print_log('NAT vs. Fake (PGGAN): AP {}'.format(plot_mi(args.save_path, 'fake2')), log)

    ens_attack(test_loader1, net, criterion, inv_factors, args, log, 20)
    if args.gpu == 0: print_log('NAT vs. ADV: AP {}'.format(plot_mi(args.save_path, 'advg')), log)


def ens_validate(val_loader, model, criterion, inv_factors, args, log, num_ens=20, suffix=''):
    model.eval()
    if args.dropout_rate > 0.:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'): m.train()

    posterior_mean = copy.deepcopy(model.state_dict())

    ece_func = _ECELoss().cuda(args.gpu)
    with torch.no_grad():
        targets = []
        mis = [0 for _ in range(len(val_loader))]
        preds = [0 for _ in range(len(val_loader))]
        rets = torch.zeros(num_ens, 9).cuda(args.gpu)

        for ens in range(num_ens):
            curvature.sample_and_replace_weights(model, inv_factors, "diag")
            for i, (input, target) in enumerate(val_loader):
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
                if ens == 0: targets.append(target)

                output = model(input)

                one_loss = criterion(output, target)

                # print(ens, i, one_loss.item())
                one_prec1, one_prec5 = accuracy(output, target, topk=(1, 5))

                mis[i] = (mis[i] * ens + (-output.softmax(-1) * output.log_softmax(-1)).sum(1)) / (ens + 1)
                preds[i] = (preds[i] * ens + output.softmax(-1)) / (ens + 1)

                loss = criterion(preds[i].log(), target)
                prec1, prec5 = accuracy(preds[i], target, topk=(1, 5))

                rets[ens, 0] += ens*target.size(0)
                rets[ens, 1] += one_loss.item()*target.size(0)
                rets[ens, 2] += one_prec1.item()*target.size(0)
                rets[ens, 3] += one_prec5.item()*target.size(0)
                rets[ens, 5] += loss.item()*target.size(0)
                rets[ens, 6] += prec1.item()*target.size(0)
                rets[ens, 7] += prec5.item()*target.size(0)

            model.load_state_dict(posterior_mean)

        preds = torch.cat(preds, 0)

        # to sync
        confidences, predictions = torch.max(preds, 1)
        targets = torch.cat(targets, 0)
        mis = (- preds * preds.log()).sum(1) - (0 if num_ens == 1 else torch.cat(mis, 0))
        rets /= targets.size(0)

        rets = rets.data.cpu().numpy()
        if suffix == '':
            ens_ece = ece_func(confidences, predictions, targets, os.path.join(args.save_path, 'ens_cal{}.pdf'.format(suffix)))
            rets[-1, -1] = ens_ece

    if args.gpu == 0:
        np.save(os.path.join(args.save_path, 'mis{}.npy'.format(suffix)), mis.data.cpu().numpy())
    return rets

def ens_attack(val_loader, model, criterion, inv_factors, args, log, num_ens=100):
    def _grad(X, y, mean, std):
        probs = torch.zeros(num_ens, X.shape[0]).cuda(args.gpu)
        grads = torch.zeros(num_ens, *list(X.shape)).cuda(args.gpu)
        for j in range(num_ens):
            with torch.enable_grad():
                X.requires_grad_()
                curvature.sample_and_replace_weights(model, inv_factors, "diag")
                output = model(X.sub(mean).div(std))
                model.load_state_dict(posterior_mean)
                loss = torch.nn.functional.cross_entropy(output, y, reduction='none')
                grad_ = torch.autograd.grad(
                    [loss], [X], grad_outputs=torch.ones_like(loss), retain_graph=False)[0].detach()
            grads[j] = grad_
            probs[j] = torch.gather(output.detach().softmax(-1), 1, y[:,None]).squeeze()
        probs /= probs.sum(0)
        grad_ = (grads * probs[:, :, None, None, None]).sum(0)
        return grad_

    def _pgd_whitebox(X, y, mean, std):
        X_pgd = X.clone()
        if args.random: X_pgd += torch.cuda.FloatTensor(*X_pgd.shape).uniform_(-args.epsilon, args.epsilon)

        for _ in range(args.num_steps):
            grad_ = _grad(X_pgd, y, mean, std)
            X_pgd += args.step_size * grad_.sign()
            eta = torch.clamp(X_pgd - X, -args.epsilon, args.epsilon)
            X_pgd = torch.clamp(X + eta, 0, 1.0)

        mis = 0
        preds = 0
        for ens in range(num_ens):
            curvature.sample_and_replace_weights(model, inv_factors, "diag")
            output = model(X_pgd.sub(mean).div(std))
            model.load_state_dict(posterior_mean)
            mis = (mis * ens + (-output.softmax(-1) * (output).log_softmax(-1)).sum(1)) / (ens + 1)
            preds = (preds * ens + output.softmax(-1)) / (ens + 1)

        loss = criterion((preds+1e-8).log(), target)
        prec1, prec5 = accuracy(preds, target, topk=(1, 5))
        mis = (- preds * (preds+1e-8).log()).sum(1) - (0 if num_ens == 1 else mis)
        return loss, prec1, prec5, mis

    if args.dataset == 'cifar10':
        mean = torch.from_numpy(np.array([x / 255 for x in [125.3, 123.0, 113.9]])).view(1,3,1,1).cuda(args.gpu).float()
        std = torch.from_numpy(np.array([x / 255 for x in [63.0, 62.1, 66.7]])).view(1,3,1,1).cuda(args.gpu).float()
    elif args.dataset == 'cifar100':
        mean = torch.from_numpy(np.array([x / 255 for x in [129.3, 124.1, 112.4]])).view(1,3,1,1).cuda(args.gpu).float()
        std = torch.from_numpy(np.array([x / 255 for x in [68.2, 65.4, 70.4]])).view(1,3,1,1).cuda(args.gpu).float()

    losses, top1, top5 = 0, 0, 0
    model.eval()
    if args.dropout_rate > 0.:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'): m.train()

    posterior_mean = copy.deepcopy(model.state_dict())

    with torch.no_grad():
        mis = []
        for i, (input, target) in enumerate(val_loader):
            print(i)
            input = input.cuda(args.gpu, non_blocking=True).mul_(std).add_(mean)
            target = target.cuda(args.gpu, non_blocking=True)
            loss, prec1, prec5, mis_ = _pgd_whitebox(input, target, mean, std)
            losses += loss * target.size(0)
            top1 += prec1 * target.size(0)
            top5 += prec5 * target.size(0)
            mis.append(mis_)

        mis = torch.cat(mis, 0)
        losses /= mis.size(0)
        top1 /= mis.size(0)
        top5 /= mis.size(0)

        if args.distributed:
            losses = reduce_tensor(losses.data, args)
            top1 = reduce_tensor(top1.data, args)
            top5 = reduce_tensor(top5.data, args)
            mis = dist_collect(mis)

    print_log('ADV ensemble TOP1: {:.4f}, TOP5: {:.4f}, LOS: {:.4f}'.format(top1.item(), top5.item(), losses.item()), log)
    if args.gpu == 0: np.save(os.path.join(args.save_path, 'mis_advg.npy'), mis.data.cpu().numpy())

def print_log(print_string, log):
    if log[1] == 0:
        print("{}".format(print_string))
        log[0].write('{}\n'.format(print_string))
        log[0].flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def load_dataset_ft(args):
    if args.dataset == 'cifar10':
        mean, std = [x / 255 for x in [125.3, 123.0, 113.9]],  [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = dset.CIFAR10
    elif args.dataset == 'cifar100':
        mean, std = [x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]]
        dataset = dset.CIFAR100

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_data = dataset(args.data_path, train=True, transform=train_transform, download=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        test_data = dataset(args.data_path, train=False, transform=test_transform, download=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data) if args.distributed else None
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=test_sampler)

        test_sampler1 = torch.utils.data.distributed.DistributedSampler(test_data) if args.distributed else None
        test_loader1 = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size//2, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=test_sampler1)

        fake_dataset = dset.ImageFolder("gan_samples/cifar10/val", test_transform)
        fake_sampler = torch.utils.data.distributed.DistributedSampler(fake_dataset) if args.distributed else None
        fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=fake_sampler)

        fake_dataset2 = dset.ImageFolder("gan_samples/cifar10/val_extra", test_transform)
        fake_sampler2 = torch.utils.data.distributed.DistributedSampler(fake_dataset2) if args.distributed else None
        fake_loader2 = torch.utils.data.DataLoader(fake_dataset2, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=fake_sampler2)

    else: assert False, 'Do not support dataset : {}'.format(args.dataset)

    return train_loader, test_loader, test_loader1, fake_loader, fake_loader2

if __name__ == '__main__': main()
