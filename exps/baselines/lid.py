from __future__ import division
import os, sys, shutil, time, random, math
import argparse
import warnings
import numpy as np
from PIL import Image
from sklearn.neighbors import KernelDensity

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import resnet_lid as models
from utils_lid import mle_batch, kmean_batch, kmean_pca_batch, score_samples, logistic_reg

sys.path.insert(0, '../')
from utils import accuracy, print_log, gaussian_kernel, smooth
from dataset.imagenet import load_dataset

parser = argparse.ArgumentParser(description='Test script for Lid and more', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', type=str,
                    default='/data/LargeData/Large/ImageNet')
parser.add_argument('--arch', metavar='ARCH', default='resnet50')
# parser.add_argument('--transferred_attack_arch', metavar='ARCH', default='resnet152')
parser.add_argument('--batch_size', type=int, default=256)

# Acceleration
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers (default: 4)')

# Random seed
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

# Adv detection
parser.add_argument('--kd_training_num', default=100000, type=int)
parser.add_argument('--kd_bandwidth', default=1., type=float) #'mnist': 3.7926, 'cifar': 0.26, 'svhn': 1.00
parser.add_argument('--kmeans_k', default=20, type=int)
parser.add_argument('--kmeans_pca', default=True, action='store_true')
parser.add_argument('--lids_k', default=20, type=int)

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

def main():
    args = parser.parse_args()
    if not os.path.isdir(args.data_path): os.makedirs(args.data_path)

    args.use_cuda = torch.cuda.is_available()
    if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.use_cuda: torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.deterministic = True

    args.gpu = 0
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    assert args.gpu is not None
    print("Use GPU: {} for training".format(args.gpu))

    # attack_net = models.__dict__[args.transferred_attack_arch](pretrained=True)
    net = models.__dict__[args.arch](pretrained=True)

    torch.cuda.set_device(args.gpu)
    net = net.cuda(args.gpu)
    # attack_net = attack_net.cuda(args.gpu)

    cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    train_loader, test_loader = load_dataset(args)

    mean = torch.from_numpy(np.array(
        [0.485, 0.456, 0.406])).view(1,3,1,1).cuda(args.gpu).float()
    std = torch.from_numpy(np.array(
        [0.229, 0.224, 0.225])).view(1,3,1,1).cuda(args.gpu).float()
    stack_kernel = gaussian_kernel().cuda(args.gpu)

    net.eval()
    # attack_net.eval()
    once_attack(train_loader, test_loader, net, None, criterion, mean, std, stack_kernel, args)

def once_attack(train_loader, val_loader, net, attack_net, criterion, mean, std, stack_kernel, args):
    def _grad(X, y, mean, std):
        with torch.enable_grad():
            X.requires_grad_()
            loss = F.cross_entropy(attack_model(X.sub(mean).div(std)), y, reduction='none')
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

    def _CW_whitebox(X, y, mean, std):
        X_cw = X.clone()
        X_cw += torch.cuda.FloatTensor(*X_cw.shape).uniform_(-args.epsilon, args.epsilon)
        y_one_hot = F.one_hot(y, num_classes=1000)
        for _ in range(args.num_steps):
            X_cw.requires_grad_()
            if X_cw.grad is not None: del X_cw.grad
            X_cw.grad = None
            with torch.enable_grad():
                logits = attack_model(X_cw.sub(mean).div(std))
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

    with torch.no_grad():
        kd_train_features = []
        kd_trian_labels = []
        for i, (input, target) in enumerate(train_loader):
            if i >= args.kd_training_num // args.batch_size: break
            # print('training,', i)
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            outputs = net(input, return_all_features = True)
            kd_train_features.append(outputs[-2].cpu().numpy())
            kd_trian_labels.append(target.cpu().numpy())
        kd_train_features = np.concatenate(kd_train_features)
        kd_trian_labels = np.concatenate(kd_trian_labels)

        kd_val_features, kd_val_labels, lids_features = {'normal': []}, {'normal': []}, {'normal': []} #kmeans_features {'normal': []}
        for attack_method in args.attack_methods:
            kd_val_features[attack_method], kd_val_labels[attack_method], lids_features[attack_method] = [], [], [] #kmeans_features[attack_method], []
        # for attack_method in args.attack_methods:
        #     if 'L2' in attack_method or 'FGSM' in attack_method or 'BIM' in attack_method:
        #         continue
        #     kd_val_features[attack_method+'_transferred'], kd_val_labels[attack_method+'_transferred'], lids_features[attack_method+'_transferred'], kmeans_features[attack_method+'_transferred'] = [], [], [], []

        losses, top1, top5 = 0, 0, 0
        for i, (input, target) in enumerate(val_loader):
            # print('testing,', i)
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            outputs = net(input, return_all_features = True)
            loss = criterion(outputs[-1], target)
            prec1, prec5 = accuracy(outputs[-1], target, topk=(1, 5))
            losses += loss * target.size(0)
            top1 += prec1 * target.size(0)
            top5 += prec5 * target.size(0)

            # for kd
            kd_val_features['normal'].append(outputs[-2].cpu().numpy())
            kd_val_labels['normal'].append(outputs[-1].argmax(dim=1).cpu().numpy())

            # for lid
            outputs.append(outputs[-1].softmax(-1))
            outputs = [output.flatten(start_dim=1).data.cpu().numpy() for output in outputs]
            lids_features['normal'].append(np.stack([mle_batch(outputs[j], outputs[j], k=args.lids_k) for j in range(len(outputs))], 1).astype(np.float32))

            # for kmeans
            # if args.kmeans_pca:
            #     kmeans_features['normal'].append(kmean_pca_batch(outputs[-2], outputs[-2], k=args.kmeans_k))
            # else:
            #     kmeans_features['normal'].append(kmean_batch(outputs[-2], outputs[-2], k=args.kmeans_k))

            input = input.mul_(std).add_(mean)
            attack_model = net
            for attack_method in args.attack_methods:
                # print(attack_method)
                X_adv = eval('_{}_whitebox'.format(attack_method))(input, target, mean, std)
                outputs_adv = net(X_adv.sub(mean).div(std), return_all_features = True)

                # for kd
                kd_val_features[attack_method].append(outputs_adv[-2].cpu().numpy())
                kd_val_labels[attack_method].append(outputs_adv[-1].argmax(dim=1).cpu().numpy())

                # for lid
                outputs_adv.append(outputs_adv[-1].softmax(-1))
                lids_features[attack_method].append(np.stack([mle_batch(outputs[j], outputs_adv[j].flatten(start_dim=1).data.cpu().numpy(), k=args.lids_k) for j in range(len(outputs))], 1).astype(np.float32))

                # for kmeans
                # if args.kmeans_pca:
                #     kmeans_features[attack_method].append(kmean_pca_batch(outputs[-2], outputs_adv[-2].flatten(start_dim=1).data.cpu().numpy(), k=args.kmeans_k))
                # else:
                #     kmeans_features[attack_method].append(kmean_batch(outputs[-2], outputs_adv[-2].flatten(start_dim=1).data.cpu().numpy(), k=args.kmeans_k))

            # attack_model = attack_net
            # for attack_method in args.attack_methods:
            #     if 'L2' in attack_method or 'FGSM' in attack_method or 'BIM' in attack_method:
            #         continue
            #     # print(attack_method + "_transferred")
            #     X_adv = eval('_{}_whitebox'.format(attack_method))(input, target, mean, std)
            #     outputs_adv = net(X_adv.sub(mean).div(std), return_all_features = True)
            #
            #     # for kd
            #     kd_val_features[attack_method+'_transferred'].append(outputs_adv[-2].cpu().numpy())
            #     kd_val_labels[attack_method+'_transferred'].append(outputs_adv[-1].argmax(dim=1).cpu().numpy())
            #
            #     # for lid
            #     outputs_adv.append(outputs_adv[-1].softmax(-1))
            #     lids_features[attack_method+'_transferred'].append(np.stack([mle_batch(outputs[j], outputs_adv[j].flatten(start_dim=1).data.cpu().numpy(), k=args.lids_k) for j in range(len(outputs))], 1).astype(np.float32))

                # for kmeans
                # if args.kmeans_pca:
                #     kmeans_features[attack_method+'_transferred'].append(kmean_pca_batch(outputs[-2], outputs_adv[-2].flatten(start_dim=1).data.cpu().numpy(), k=args.kmeans_k))
                # else:
                #     kmeans_features[attack_method+'_transferred'].append(kmean_batch(outputs[-2], outputs_adv[-2].flatten(start_dim=1).data.cpu().numpy(), k=args.kmeans_k))

        losses /= len(val_loader.dataset)
        top1 /= len(val_loader.dataset)
        top5 /= len(val_loader.dataset)

    print('TOP1: {:.4f}, TOP5: {:.4f}, LOS: {:.4f}'.format(top1.item(), top5.item(), losses.item()))

    for map_ in [kd_val_features, kd_val_labels, lids_features]:#, kmeans_features]:
        for k in map_.keys():
            map_[k] = np.concatenate(map_[k])
            print(k, map_[k].shape)

    # for kd
    print('Training KDEs...')
    kdes = {i: None for i in range(1000)}
    for i in range(1000):
        tmp = kd_trian_labels == i
        if tmp.sum() > 0:
            kdes[i] = KernelDensity(kernel='gaussian',
                                    bandwidth=args.kd_bandwidth) \
                .fit(kd_train_features[np.where(tmp)])

    print('computing densities...')
    kd_densities = {}
    for k in kd_val_features.keys():
        kd_densities[k] = score_samples(
            kdes,
            kd_val_features[k],
            kd_val_labels[k]
        )

    for k in kd_densities.keys():
        np.save("/data/zhijie/snapshots_ba/lids/kd_{}.npy".format(k), kd_densities[k])
        # if k != 'normal':
            # print("-------------KD normal vs. {}-------------".format(k))
            # logistic_reg(kd_densities['normal'], kd_densities[k])

    # for lid
    for k in lids_features.keys():
        np.save("/data/zhijie/snapshots_ba/lids/lid_{}.npy".format(k), lids_features[k])
        # if k != 'normal':
        #     print("-------------LiD normal vs. {}-------------".format(k))
        #     logistic_reg(lids_features['normal'], lids_features[k])

    # for kmeans
    # for k in kmeans_features.keys():
    #     if k != 'normal':
    #         print("-------------Kmeans normal vs. {}-------------".format(k))
    #         logistic_reg(kmeans_features['normal'], kmeans_features[k])


if __name__ == '__main__': main()
