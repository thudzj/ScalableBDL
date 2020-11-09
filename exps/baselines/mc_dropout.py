from __future__ import division
import os, sys, shutil, time, random, math
import argparse
import warnings
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.optim import SGD
import torch.backends.cudnn as cudnn

import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.nn.functional as F

sys.path.insert(0, '../')
from utils import plot_mi, accuracy, \
    reduce_tensor, dist_collect, print_log, \
    gaussian_kernel, smooth
from dataset.imagenet import load_dataset_ft
import models.resnet as models
from models.resnet import Bottleneck
from models.utils import load_state_dict_from_url


parser = argparse.ArgumentParser(description='Test script for MC dropout', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', type=str,
                    default='/data/LargeData/Large/ImageNet')
parser.add_argument('--dataset', metavar='DSET', type=str, default='imagenet')
parser.add_argument('--arch', metavar='ARCH', default='resnet50')
parser.add_argument('--transferred_attack_arch', metavar='ARCH', default='resnet152')
parser.add_argument('--batch_size', type=int, default=256)

# Checkpoints
parser.add_argument('--save_path', type=str, default='/data/zhijie/snapshots_ba/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--job-id', type=str, default='mc_dropout')

# Acceleration
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers (default: 4)')

# Random seed
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

# Bayesian
parser.add_argument('--unc_metric', type=str, default='feature_var')
parser.add_argument('--num_mc_samples', type=int, default=20)

# Attack settings
parser.add_argument('--attack_methods', type=str, nargs='+',
                    default=['FGSM', 'CW', 'BIM', 'PGD', 'MIM', 'TIM', 'DI_MIM', 'FGSM_L2', 'BIM_L2', 'PGD_L2'])
parser.add_argument('--mim_momentum', default=1., type=float,
                    help='mim_momentum')
parser.add_argument('--epsilon', default=16./255., type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=1./255., type=float,
                    help='perturb step size')
parser.add_argument('--random', default=True,
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

    log = open(os.path.join(args.save_path, 'log_seed{}.txt'.format(args.manualSeed)), 'w')
    log = (log, args.gpu)

    attack_net = models.__dict__[args.transferred_attack_arch](pretrained=True)

    net = models.__dict__[args.arch](pretrained=False)
    for module in net.modules():
        if isinstance(module, Bottleneck):
            setattr(module, 'conv2', torch.nn.Sequential(module.conv2, torch.nn.Dropout(p=0.2)))
            setattr(module, 'conv3', torch.nn.Sequential(module.conv3, torch.nn.Dropout(p=0.2)))
    net.load_state_dict(load_state_dict_from_url(
        'http://ml.cs.tsinghua.edu.cn/~zhijie/files/resnet50-dropout0.2-afterbn.pth', # to do: correct the name
        progress=True))

    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch  version : {}".format(torch.__version__), log)
    print_log("CuDNN  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Number of parameters: {}".format(sum([p.numel() for p in net.parameters()])), log)
    print_log(str(args), log)

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url+":"+args.dist_port,
                                world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.gpu)
        net.cuda(args.gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])

        attack_net.cuda(args.gpu)
        attack_net = torch.nn.parallel.DistributedDataParallel(attack_net, device_ids=[args.gpu])
    else:
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)
        attack_net = attack_net.cuda(args.gpu)

    cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    train_loader, test_loader, _ = load_dataset_ft(args)

    mean = torch.from_numpy(np.array(
        [0.485, 0.456, 0.406])).view(1,3,1,1).cuda(args.gpu).float()
    std = torch.from_numpy(np.array(
        [0.229, 0.224, 0.225])).view(1,3,1,1).cuda(args.gpu).float()
    stack_kernel = gaussian_kernel().cuda(args.gpu)

    evaluate(test_loader, net, attack_net, criterion, mean, std, stack_kernel, args, log)
    log[0].close()

def evaluate(test_loader, net, attack_net, criterion, mean, std, stack_kernel, args, log):

    top1, top5, val_loss = ens_validate(test_loader, net, criterion, args, log)
    print_log('Parallel ensemble {} TOP1: {:.4f}, TOP5: {:.4f}, LOS: {:.4f}'.format(args.num_mc_samples, top1, top5, val_loss), log)

    for attack_method in args.attack_methods:
        ens_attack(test_loader, net, criterion, mean, std, stack_kernel, args, log, attack_method)
        if args.gpu == 0:
            print_log('NAT vs. {} --> {}'.format(attack_method, plot_mi(args.save_path, attack_method)), log)

    for attack_method in args.attack_methods:
        if 'L2' in attack_method or 'FGSM' in attack_method or 'BIM' in attack_method:
            continue
        ens_attack(test_loader, net, criterion, mean, std, stack_kernel, args, log, attack_method, attack_model=attack_net)
        if args.gpu == 0:
            print_log('NAT vs. {} from {} --> {}'.format(attack_method + "_transferred", args.transferred_attack_arch, plot_mi(args.save_path, attack_method + "_transferred")), log)

    return top1, val_loss

def ens_validate(val_loader, model, criterion, args, log, suffix=''):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'): m.train()

    with torch.no_grad():
        mis = []
        top1, top5, val_loss = 0, 0, 0
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            features, outputs = [], []
            for _ in range(args.num_mc_samples):
                feature, output = model(input, return_features=True)
                features.append(feature)
                outputs.append(output)
            features = torch.stack(features, 1)
            outputs = torch.stack(outputs, 1)
            assert outputs.dim() == 3
            output = outputs.softmax(-1).mean(-2)
            if args.unc_metric == 'feature_var':
                mi = features.var(dim=1).mean(dim=[1,2,3])
            elif args.unc_metric == 'softmax_var':
                mi = outputs.softmax(-1).var(dim=1).mean(dim=[1])
            else:
                mi = (- output * output.log()).sum(1) - (- outputs.softmax(-1) * outputs.log_softmax(-1)).sum(2).mean(1)
            mis.append(mi)
            loss = criterion((output+1e-10).log(), target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            top1 += prec1*target.size(0)
            top5 += prec5*target.size(0)
            val_loss += loss*target.size(0)

        mis = torch.cat(mis, 0)
        top1 /= mis.size(0)
        top5 /= mis.size(0)
        val_loss /= mis.size(0)

        # to sync
        if args.distributed:
            mis = dist_collect(mis)
            top1 = reduce_tensor(top1.data, args)
            top5 = reduce_tensor(top5.data, args)
            val_loss = reduce_tensor(val_loss.data, args)

    if args.gpu == 0:
        np.save(os.path.join(args.save_path, 'mis{}.npy'.format(suffix)), mis.data.cpu().numpy())
    return top1.item(), top5.item(), val_loss.item()

def ens_attack(val_loader, model, criterion, mean, std, stack_kernel, args, log, attack_method, attack_model=None):
    def _grad(X, y, mean, std):
        if attack_model == model:
            probs = torch.zeros(args.num_mc_samples, X.shape[0]).cuda(args.gpu)
            grads = torch.zeros(args.num_mc_samples, *list(X.shape)).cuda(args.gpu)
            for j in range(args.num_mc_samples):
                with attack_model.no_sync():
                    with torch.enable_grad():
                        X.requires_grad_()
                        output = attack_model(X.sub(mean).div(std))
                        loss = torch.nn.functional.cross_entropy(output, y, reduction='none')
                        grad_ = torch.autograd.grad(
                            [loss], [X], grad_outputs=torch.ones_like(loss), retain_graph=False)[0].detach()
                grads[j] = grad_
                probs[j] = torch.gather(output.detach().softmax(-1), 1, y[:,None]).squeeze()
            probs /= probs.sum(0)
            grad_ = (grads * probs[:, :, None, None, None]).sum(0)
        else:
            with attack_model.no_sync():
                with torch.enable_grad():
                    X.requires_grad_()
                    outputs = attack_model(X.sub(mean).div(std)).softmax(-1)
                    if outputs.dim() == 3:
                        output = outputs.mean(-2) + 1e-10
                    else:
                        output = outputs
                    loss = F.cross_entropy(output.log(), y, reduction='none')
                    grad_ = torch.autograd.grad(
                        [loss], [X], grad_outputs=torch.ones_like(loss),
                        retain_graph=False)[0].detach()
        return grad_

    def _PGD_whitebox(X, y, mean, std):
        X_pgd = X.clone()
        if args.random:
            X_pgd += torch.cuda.FloatTensor(*X_pgd.shape).uniform_(-args.epsilon, args.epsilon)
        for _ in range(args.num_steps):
            grad_ = _grad(X_pgd, y, mean, std)
            X_pgd += args.step_size * grad_.sign()
            eta = torch.clamp(X_pgd - X, -args.epsilon, args.epsilon)
            X_pgd = torch.clamp(X + eta, 0, 1.0)
        return X_pgd

    def _PGD_L2_whitebox(X, y, mean, std):
        bs = X.shape[0]
        scale_ = np.sqrt(np.prod(list(X.shape[1:])))
        lr = args.step_size * scale_
        radius = args.epsilon * scale_

        X_pgd = X.clone()
        if args.random:
            X_pgd += torch.cuda.FloatTensor(*X_pgd.shape).uniform_(-args.epsilon, args.epsilon)
        for _ in range(args.num_steps):
            grad_ = _grad(X_pgd, y, mean, std)
            grad_norm_ = torch.clamp(torch.norm(grad_.view(bs, -1), dim=1), min=1e-12).view(bs, 1, 1, 1)
            grad_unit_ = grad_ / grad_norm_
            X_pgd += lr * grad_unit_

            eta = X_pgd - X
            eta_norm = torch.clamp(torch.norm(eta.view(bs, -1), dim=1), min=radius).view(bs, 1, 1, 1)
            eta = eta * (radius / eta_norm)
            X_pgd = torch.clamp(X + eta, 0, 1.0)
        return X_pgd

    def _FGSM_whitebox(X, y, mean, std):
        X_fgsm = X.clone()
        grad_ = _grad(X_fgsm, y, mean, std)
        eta = args.epsilon * grad_.sign()
        X_fgsm = torch.clamp(X_fgsm + eta, 0, 1.0)
        return X_fgsm

    def _FGSM_RS_whitebox(X, y, mean, std):
        X_fgsm = X.clone()
        X_fgsm += torch.cuda.FloatTensor(*X_fgsm.shape).uniform_(-args.epsilon, args.epsilon)
        grad_ = _grad(X_fgsm, y, mean, std)
        eta = args.epsilon * grad_.sign()
        X_fgsm = torch.clamp(X_fgsm + eta, 0, 1.0)
        return X_fgsm

    def _FGSM_L2_whitebox(X, y, mean, std):
        X_fgsm = X.clone()
        grad_ = _grad(X_fgsm, y, mean, std)
        grad_norm_ = torch.clamp(torch.norm(grad_.view(X.shape[0], -1), dim=1), min=1e-12).view(X.shape[0], 1, 1, 1)
        grad_unit_ = grad_ / grad_norm_
        eta = args.epsilon * np.sqrt(np.prod(list(X.shape[1:]))) * grad_unit_
        X_fgsm = torch.clamp(X_fgsm + eta, 0, 1.0)
        return X_fgsm

    def _BIM_whitebox(X, y, mean, std):
        X_bim = X.clone()
        for _ in range(args.num_steps):
            grad_ = _grad(X_bim, y, mean, std)
            X_bim += args.step_size * grad_.sign()
            eta = torch.clamp(X_bim - X, -args.epsilon, args.epsilon)
            X_bim = torch.clamp(X + eta, 0, 1.0)
        return X_bim

    def _BIM_L2_whitebox(X, y, mean, std):
        bs = X.shape[0]
        scale_ = np.sqrt(np.prod(list(X.shape[1:])))
        lr = args.step_size * scale_
        radius = args.epsilon * scale_

        X_bim = X.clone()
        for _ in range(args.num_steps):
            grad_ = _grad(X_bim, y, mean, std)
            grad_norm_ = torch.clamp(torch.norm(grad_.view(bs, -1), dim=1), min=1e-12).view(bs, 1, 1, 1)
            grad_unit_ = grad_ / grad_norm_
            X_bim += lr * grad_unit_

            eta = X_bim - X
            eta_norm = torch.clamp(torch.norm(eta.view(bs, -1), dim=1), min=radius).view(bs, 1, 1, 1)
            eta = eta * (radius / eta_norm)
            X_bim = torch.clamp(X + eta, 0, 1.0)
        return X_bim

    def _MIM_whitebox(X, y, mean, std):
        X_mim = X.clone()
        g = torch.zeros_like(X_mim)
        for _ in range(args.num_steps):
            grad_ = _grad(X_mim, y, mean, std)
            grad_ /= grad_.abs().mean(dim=[1,2,3], keepdim=True)
            g = g * args.mim_momentum + grad_
            X_mim += args.step_size * g.sign()
            eta = torch.clamp(X_mim - X, -args.epsilon, args.epsilon)
            X_mim = torch.clamp(X + eta, 0, 1.0)
        return X_mim

    def _TIM_whitebox(X, y, mean, std):
        X_tim = X.clone()
        g = torch.zeros_like(X_tim)
        for _ in range(args.num_steps):
            grad_ = _grad(X_tim, y, mean, std)
            grad_ = smooth(grad_, stack_kernel)
            grad_ /= grad_.abs().mean(dim=[1,2,3], keepdim=True)
            g = g * args.mim_momentum + grad_
            X_tim += args.step_size * g.sign()
            eta = torch.clamp(X_tim - X, -args.epsilon, args.epsilon)
            X_tim = torch.clamp(X + eta, 0, 1.0)
        return X_tim

    # def _MIM_L2_whitebox(X, y, mean, std):
    #     bs = X.shape[0]
    #     scale_ = np.sqrt(np.prod(list(X.shape[1:])))
    #     lr = args.step_size * scale_
    #     radius = args.epsilon * scale_
    #
    #     X_mim = X.clone()
    #     g = torch.zeros_like(X_mim)
    #     for _ in range(args.num_steps):
    #         grad_ = _grad(X_mim, y, mean, std)
    #
    #     return X_mim

    def _CW_whitebox(X, y, mean, std):
        X_cw = X.clone()
        X_cw += torch.cuda.FloatTensor(*X_cw.shape).uniform_(-args.epsilon, args.epsilon)
        y_one_hot = F.one_hot(y, num_classes=1000)
        for _ in range(args.num_steps):
            X_cw.requires_grad_()
            if X_cw.grad is not None: del X_cw.grad
            X_cw.grad = None
            with torch.enable_grad():
                outputs = attack_model(X_cw.sub(mean).div(std)).softmax(-1)
                if outputs.dim() == 3:
                    logits = (outputs.mean(-2) + 1e-10).log()
                else:
                    logits = outputs.log()
                logit_target = torch.max(y_one_hot * logits, 1)[0]
                logit_other = torch.max(
                    (1 - y_one_hot) * logits - 1e6 * y_one_hot, 1)[0]
                loss = torch.mean(logit_other - logit_target)
                loss.backward()

            X_cw += args.step_size * X_cw.grad.sign()
            eta = torch.clamp(X_cw - X, -args.epsilon, args.epsilon)
            X_cw = torch.clamp(X + eta, 0, 1.0)
        return X_cw

    def _DI_MIM_whitebox(X, y, mean, std):
        def Resize_and_padding(x, scale_factor=1.1):
            ori_size = x.size(-1)
            new_size = int(x.size(-1) * scale_factor)
            delta_w = new_size - ori_size
            delta_h = new_size - ori_size
            top = random.randint(0, delta_h)
            left = random.randint(0, delta_w)
            bottom = delta_h - top
            right = delta_w - left
            x = F.pad(x, pad=(left,right,top,bottom), value=0)
            return F.interpolate(x, size = ori_size)

        X_mim = X.clone()
        g = torch.zeros_like(X_mim)
        for _ in range(args.num_steps):
            grad_ = _grad(Resize_and_padding(X_mim), y, mean, std)
            grad_ /= grad_.abs().mean(dim=[1,2,3], keepdim=True)
            g = g * args.mim_momentum + grad_
            X_mim += args.step_size * g.sign()
            eta = torch.clamp(X_mim - X, -args.epsilon, args.epsilon)
            X_mim = torch.clamp(X + eta, 0, 1.0)
        return X_mim

    is_transferred = True if (attack_model is not None and attack_model != model) else False
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'): m.train()

    if is_transferred:
        attack_model.eval()
        for m in attack_model.modules():
            if m.__class__.__name__.startswith('Dropout'): m.train()
    else:
        attack_model = model

    with torch.no_grad():
        losses, top1, top5, num_data = 0, 0, 0, 0
        mis = []
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True).mul_(std).add_(mean)
            target = target.cuda(args.gpu, non_blocking=True)

            X_adv = eval('_{}_whitebox'.format(attack_method))(input, target, mean, std)

            features, outputs = [], []
            for _ in range(args.num_mc_samples):
                feature, output = model(X_adv.sub(mean).div(std), return_features=True)
                features.append(feature)
                outputs.append(output)
            features = torch.stack(features, 1)
            outputs = torch.stack(outputs, 1)

            output = outputs.softmax(-1).mean(-2)
            if args.unc_metric == 'feature_var':
                mi = features.var(dim=1).mean(dim=[1,2,3])
            elif args.unc_metric == 'softmax_var':
                mi = outputs.softmax(-1).var(dim=1).mean(dim=[1])
            else:
                mi = (- output * output.log()).sum(1) - (- outputs.softmax(-1) * outputs.log_softmax(-1)).sum(2).mean(1)
            mis.append(mi)
            loss = criterion((output+1e-10).log(), target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            losses += loss * target.size(0)
            top1 += prec1 * target.size(0)
            top5 += prec5 * target.size(0)
            num_data += target.size(0)

        losses /= num_data
        top1 /= num_data
        top5 /= num_data
        losses = reduce_tensor(losses.data, args)
        top1 = reduce_tensor(top1.data, args)
        top5 = reduce_tensor(top5.data, args)

        mis = torch.cat(mis, 0)
        if args.distributed: mis = dist_collect(mis)

    # print_log('Attack by {}, ensemble TOP1: {:.4f}, TOP5: {:.4f}, LOS: {:.4f}'.format(
    #     attack_method, top1.item(), top5.item(), losses.item()), log)
    if args.gpu == 0 and mis is not None:
        np.save(os.path.join(args.save_path, 'mis_{}{}.npy'.format(attack_method, "_transferred" if is_transferred else "")), mis.data.cpu().numpy())

if __name__ == '__main__': main()
