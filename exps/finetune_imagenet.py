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

from kornia import gaussian_blur2d

from utils import AverageMeter, RecorderMeter, time_string, \
    convert_secs2time, _ECELoss, plot_mi, accuracy, \
    reduce_tensor, dist_collect, print_log, save_checkpoint, \
    gaussian_kernel, smooth
from dataset.imagenet import load_dataset_ft
import models.resnet as models
# from models.resnet import conv1x1

# step 0: clone the ScalableBDL repo and checkout to the efficient branch
sys.path.insert(0, '../')
from scalablebdl.mean_field import PsiSGD
from scalablebdl.mean_field import to_bayesian as to_bayesian_mfg
from scalablebdl.empirical import to_bayesian as to_bayesian_emp
from scalablebdl.bnn_utils import freeze, unfreeze, set_mc_sample_id, \
    disable_dropout, parallel_eval, disable_parallel_eval

parser = argparse.ArgumentParser(description='Training script for ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', type=str,
                    default='/data/LargeData/Large/ImageNet')
parser.add_argument('--dataset', metavar='DSET', type=str, default='imagenet')
parser.add_argument('--arch', metavar='ARCH', default='resnet50')
parser.add_argument('--transferred_attack_arch', metavar='ARCH', default='resnet152')

# Optimization
parser.add_argument('--epochs', metavar='N', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--ft_lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)

#Regularization
parser.add_argument('--decay', type=float, default=1e-4,
                    help='Weight decay')

# Checkpoints
parser.add_argument('--save_path', type=str, default='/data/zhijie/snapshots_ba/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--job-id', type=str, default='onelayer-imagenet')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='Evaluate model on test set')
parser.add_argument('--inference_speed_profile', type=str, default=None)

# Acceleration
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers (default: 4)')

# Random seed
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

# Bayesian
parser.add_argument('--unc_metric', type=str, default='feature_var')
parser.add_argument('--posterior_type', type=str, default='mfg')
parser.add_argument('--psi_init_range', type=float, nargs='+', default=[-6, -5])
parser.add_argument('--alpha', type=float, default=1.)
parser.add_argument('--num_mc_samples', type=int, default=20)
parser.add_argument('--uncertainty_threshold', type=float, default=0.5)
parser.add_argument('--epsilon_max_train', type=float, default=2.)
parser.add_argument('--epsilon_min_train', type=float, default=1.)
parser.add_argument('--blur_prob', type=float, default=0.1)
parser.add_argument('--blur_kernel_size', type=int, nargs='+', default=[3, 5, 7, 9, 11])
parser.add_argument('--blur_sig', type=float, nargs='+', default=[0., 5.])
parser.add_argument('--batchwise_sample', action='store_true', default=False)
parser.add_argument('--full', action='store_true', default=False)

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
parser.add_argument('--mimicry', default=0., type=float)

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

    # step 0: note to pre-process some hyper-paramters
    args.epsilon_max_train *= args.epsilon
    args.epsilon_min_train *= args.epsilon

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

    if args.inference_speed_profile != None:
        net = models.__dict__[args.arch](pretrained=True)
        if args.inference_speed_profile == 'libre':
            net.layer4[-1] = to_bayesian_emp(net.layer4[-1], args.num_mc_samples)
        elif args.inference_speed_profile == 'LMFVI':
            net.layer4[-1] = to_bayesian_mfg(net.layer4[-1], args.psi_init_range, args.num_mc_samples)
        elif args.inference_speed_profile == 'MFVI':
            net = to_bayesian_mfg(net, args.psi_init_range, args.num_mc_samples)
        elif args.inference_speed_profile == 'MC_dropout':
            from models.resnet import Bottleneck
            for module in net.modules():
                if isinstance(module, Bottleneck):
                    setattr(module, 'conv2', torch.nn.Sequential(module.conv2, torch.nn.Dropout(p=0.2)))
                    setattr(module, 'conv3', torch.nn.Sequential(module.conv3, torch.nn.Dropout(p=0.2)))
        elif args.inference_speed_profile == 'DNN':
            pass
        else:
            raise NotImplementedError
        net.cuda(args.gpu)
        net.eval()
        if args.inference_speed_profile == 'LMFVI' or args.inference_speed_profile == 'libre':
            parallel_eval(net)
        if args.inference_speed_profile == 'MC_dropout':
            for m in net.modules():
                if m.__class__.__name__.startswith('Dropout'): m.train()

        cudnn.benchmark = True

        dummy_input = torch.zeros(32, 3, 224, 224).cuda(args.gpu)

        with torch.no_grad():
            for i in range(200):
                if i == 100:
                    time0 = time.time()
                if args.inference_speed_profile == 'LMFVI' or args.inference_speed_profile == 'libre' or args.inference_speed_profile== 'DNN':
                    outputs = net(dummy_input)
                else:
                    outputs = []
                    for j in range(args.num_mc_samples):
                        outputs.append(net(dummy_input))
                    outputs = torch.stack(outputs, 1)
                if args.inference_speed_profile== 'DNN':
                    assert(outputs.dim() == 2 and outputs.size(0) == 32)
                else:
                    assert(outputs.dim() == 3 and outputs.size(1) == args.num_mc_samples)
        time1 = time.time()
        print((time1 - time0) / 100.)
        exit()

    log = open(os.path.join(args.save_path, 'log_seed{}{}.txt'.format(
               args.manualSeed, '_eval' if args.evaluate else '')), 'w')
    log = (log, args.gpu)

    attack_net = models.__dict__[args.transferred_attack_arch](pretrained=True)

    # step 1: convert the last res-block to be Bayesian
    net = models.__dict__[args.arch](pretrained=True)
    # disable_dropout(net) # we may need to disable the dropout (not sure, need try)
    if args.posterior_type == 'mfg':
        if args.full:
            net = to_bayesian_mfg(net, args.psi_init_range, args.num_mc_samples)
        else:
            net.layer4[-1].conv1 = to_bayesian_mfg(net.layer4[-1].conv1, args.psi_init_range, args.num_mc_samples)
            net.layer4[-1].bn1 = to_bayesian_mfg(net.layer4[-1].bn1, args.psi_init_range, args.num_mc_samples)
            net.layer4[-1].conv2 = to_bayesian_mfg(net.layer4[-1].conv2, args.psi_init_range, args.num_mc_samples)
            net.layer4[-1].bn2 = to_bayesian_mfg(net.layer4[-1].bn2, args.psi_init_range, args.num_mc_samples)
            net.layer4[-1].conv3 = to_bayesian_mfg(net.layer4[-1].conv3, args.psi_init_range, args.num_mc_samples)
            net.layer4[-1].bn3 = to_bayesian_mfg(net.layer4[-1].bn3, args.psi_init_range, args.num_mc_samples)
            # assert(net.layer4[-1].conv1.in_channels == net.layer4[-1].bn3.num_features)
            # net.layer4[-1].downsample = to_bayesian_mfg(torch.nn.Sequential(
            #         conv1x1(net.layer4[-1].conv1.in_channels, net.layer4[-1].bn3.num_features, 1),
            #         torch.nn.BatchNorm2d(net.layer4[-1].bn3.num_features),
            #     ), args.psi_init_range, args.num_mc_samples, is_residual=True)
    elif args.posterior_type == 'emp':
        net.layer4[-1].conv1 = to_bayesian_emp(net.layer4[-1].conv1, args.num_mc_samples)
        net.layer4[-1].bn1 = to_bayesian_emp(net.layer4[-1].bn1, args.num_mc_samples)
        net.layer4[-1].conv2 = to_bayesian_emp(net.layer4[-1].conv2, args.num_mc_samples)
        net.layer4[-1].bn2 = to_bayesian_emp(net.layer4[-1].bn2, args.num_mc_samples)
        net.layer4[-1].conv3 = to_bayesian_emp(net.layer4[-1].conv3, args.num_mc_samples)
        net.layer4[-1].bn3 = to_bayesian_emp(net.layer4[-1].bn3, args.num_mc_samples)
        # assert(net.layer4[-1].conv1.in_channels == net.layer4[-1].bn3.num_features)
        # net.layer4[-1].downsample = to_bayesian_emp(torch.nn.Sequential(
        #         conv1x1(net.layer4[-1].conv1.in_channels, net.layer4[-1].bn3.num_features, 1),
        #         torch.nn.BatchNorm2d(net.layer4[-1].bn3.num_features),
        #     ), args.num_mc_samples, is_residual=True)
    else:
        raise NotImplementedError

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

    # step 2: build optimizers for the pre-trianed and added params
    pre_trained, new_added = [], []
    for name, param in net.named_parameters():
        if (args.posterior_type == 'mfg' and 'psi' in name) \
                or (args.posterior_type == 'emp' and ('weights' in name or 'biases' in name)):
            new_added.append(param)
        else:
            pre_trained.append(param)
    pre_trained_optimizer = SGD(pre_trained, args.ft_lr, args.momentum,
                                weight_decay=args.decay)

    if args.posterior_type == 'mfg':
        new_added_optimizer = PsiSGD(new_added, args.lr, args.momentum,
                                     weight_decay=args.decay)
    else:
        new_added_optimizer = SGD(new_added, args.lr, args.momentum,
                                  weight_decay=args.decay/args.num_mc_samples)

    recorder = RecorderMeter(args.epochs)
    if args.resume:
        if args.resume == 'auto':
            args.resume = os.path.join(args.save_path, 'checkpoint.pth.tar')
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpu))
            recorder = checkpoint['recorder']
            recorder.refresh(args.epochs)
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'] if args.distributed
                else {k.replace('module.', ''): v for k,v in checkpoint['state_dict'].items()})
            pre_trained_optimizer.load_state_dict(checkpoint['pre_trained_optimizer'])
            new_added_optimizer.load_state_dict(checkpoint['new_added_optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})".format(
                args.resume, best_acc, checkpoint['epoch']), log)
            del checkpoint
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for the model", log)

    cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    train_loader, test_loader, sub_test_loader = load_dataset_ft(args)

    if args.dataset == 'cifar10':
        mean = torch.from_numpy(np.array([x / 255
            for x in [125.3, 123.0, 113.9]])).view(1,3,1,1).cuda(args.gpu).float()
        std = torch.from_numpy(np.array([x / 255
            for x in [63.0, 62.1, 66.7]])).view(1,3,1,1).cuda(args.gpu).float()
    elif args.dataset == 'cifar100':
        mean = torch.from_numpy(np.array([x / 255
            for x in [129.3, 124.1, 112.4]])).view(1,3,1,1).cuda(args.gpu).float()
        std = torch.from_numpy(np.array([x / 255
            for x in [68.2, 65.4, 70.4]])).view(1,3,1,1).cuda(args.gpu).float()
    elif args.dataset == 'imagenet':
        mean = torch.from_numpy(np.array(
            [0.485, 0.456, 0.406])).view(1,3,1,1).cuda(args.gpu).float()
        std = torch.from_numpy(np.array(
            [0.229, 0.224, 0.225])).view(1,3,1,1).cuda(args.gpu).float()
    stack_kernel = gaussian_kernel().cuda(args.gpu)

    # step 2: set the num_data arg for new_added_optimizer if using mean-field Gaussian
    if args.posterior_type == 'mfg':
        new_added_optimizer.num_data = len(train_loader.dataset)

    if args.evaluate:
        while True:
            evaluate(test_loader, sub_test_loader,
                     net, attack_net, criterion, mean, std, stack_kernel, args, log)
        return

    start_time = time.time()
    epoch_time = AverageMeter()
    train_los = -1

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(
                                    time_string(), epoch, args.epochs, need_time) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        train_acc, train_los = train(train_loader, net,
                                     criterion, mean, std, stack_kernel, pre_trained_optimizer, new_added_optimizer,
                                     epoch, args, log)
        if epoch == args.epochs - 1 or args.full:
            val_acc, val_los = 0, 0
        else:
            val_acc, val_los = evaluate(test_loader, sub_test_loader,
                                    net, attack_net, criterion, mean, std, stack_kernel, args, log, sub_attack=True)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        if args.gpu == 0:
            save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': net.state_dict(),
              'recorder': recorder,
              'pre_trained_optimizer' : pre_trained_optimizer.state_dict(),
              'new_added_optimizer' : new_added_optimizer.state_dict(),
            }, False, args.save_path, 'checkpoint.pth.tar')

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'log.png'))

    while True:
        evaluate(test_loader, sub_test_loader,
                 net, attack_net, criterion, mean, std, stack_kernel, args, log)

    log[0].close()


def train(train_loader, model, criterion, mean, std, stack_kernel,
          pre_trained_optimizer, new_added_optimizer, epoch, args, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ur_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        slr = adjust_learning_rate_per_step(new_added_optimizer, epoch, i, len(train_loader), args)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        bs = input.shape[0]

        # step 3: fetch the half of the batch for generating uniformly perturbed samples
        bs1 = bs // 2
        eps = np.random.uniform(args.epsilon_min_train, args.epsilon_max_train)
        input1 = input[:bs1].mul(std).add(mean)
        noise = torch.empty_like(input1).uniform_(-eps, eps)

        gaussian_filter_kernel_size = int(np.random.choice(args.blur_kernel_size))
        gaussian_filter_sigma = np.random.uniform(args.blur_sig[0], args.blur_sig[1])
        noise_lf = gaussian_blur2d(noise, (gaussian_filter_kernel_size, gaussian_filter_kernel_size), (gaussian_filter_sigma, gaussian_filter_sigma))

        # step 3: using low-frequency noise with probability blur_prob
        mask = (torch.empty(bs1, device=input.device).random_(100) <
            args.blur_prob * 100).float().view(-1, 1, 1, 1)
        noise = noise_lf * mask + noise * (1 - mask)

        # step 3: apply the noise
        input1 = noise.add_(input1).clamp_(0, 1).sub_(mean).div_(std)

        # step 4: set mc_sample_id if using empirical distribution (population based)
        if args.posterior_type == 'emp':
            if args.batchwise_sample: #for not using uncertainty reg. only
                tmp = np.random.choice(args.num_mc_samples, 2, replace=False)
                mc_sample_id = np.concatenate([np.random.randint(0, args.num_mc_samples) * np.ones(bs), tmp[0] * np.ones(bs1), tmp[1] * np.ones(bs1)])
            else:
                mc_sample_id = np.concatenate([np.random.randint(0, args.num_mc_samples, size=bs),
                    np.stack([np.random.choice(args.num_mc_samples, 2, replace=False) for _ in range(bs1)]).T.reshape(-1)])
            set_mc_sample_id(model, args.num_mc_samples, mc_sample_id)

        # step 5: forward prop for predicted logits of normal data and features of noisy data
        features, output = model(torch.cat([input, input1.repeat(2, 1, 1, 1)]), return_features=True)

        # step 6: calculate CE loss and uncertainty loss
        loss = criterion(output[:bs], target)
        out1_0 = features[bs:bs+bs1]
        out1_1 = features[bs+bs1:]
        mi = ((out1_0 - out1_1)**2).mean(dim=[1,2,3])
        ur_loss = F.relu(args.uncertainty_threshold - mi).mean()

        prec1, prec5 = accuracy(output[:bs], target, topk=(1, 5))
        losses.update(loss.detach().item(), bs)
        ur_losses.update(ur_loss.detach().item(), bs1)
        top1.update(prec1.item(), bs)
        top5.update(prec5.item(), bs)

        # step 7: backward prop and perform optimization
        pre_trained_optimizer.zero_grad()
        new_added_optimizer.zero_grad()
        (loss+ur_loss*args.alpha).backward()
        pre_trained_optimizer.step()
        new_added_optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i == len(train_loader) - 1:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                        'Time {batch_time.avg:.3f}   '
                        'Data {data_time.avg:.3f}   '
                        'Loss {loss.avg:.4f}   '
                        'UR Loss {ur_loss.avg:.4f}   '
                        'Prec@1 {top1.avg:.3f}   '
                        'Prec@5 {top5.avg:.3f}   '
                        'EPS {eps:.4f}   '
                        'LR {slr:.4f}   '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        ur_loss=ur_losses, data_time=data_time, loss=losses,
                        top1=top1, top5=top5, eps=eps, slr=slr) + time_string(), log)
    return top1.avg, losses.avg


def evaluate(test_loader, sub_test_loader,
             net, attack_net, criterion, mean, std, stack_kernel, args, log, sub_attack=False):

    top1, top5, val_loss, ens_ece = ens_validate(test_loader, net, criterion, args, log)
    print_log('Parallel ensemble {} TOP1: {:.4f}, TOP5: {:.4f}, LOS: {:.4f},'
              ' ECE: {:.4f}'.format(args.num_mc_samples, top1, top5, val_loss, ens_ece), log)

    for attack_method in args.attack_methods:
        ens_attack(sub_test_loader if sub_attack else test_loader,
                   net, criterion, mean, std, stack_kernel, args, log, attack_method)
        if args.gpu == 0:
            print_log('NAT vs. {} --> {}'.format(attack_method, plot_mi(args.save_path, attack_method)), log)

    if args.mimicry == 0:
        for attack_method in args.attack_methods:
            if 'L2' in attack_method or 'FGSM' in attack_method or 'BIM' in attack_method:
                continue
            ens_attack(sub_test_loader if sub_attack else test_loader,
                       net, criterion, mean, std, stack_kernel, args, log, attack_method, attack_model=attack_net)
            if args.gpu == 0:
                print_log('NAT vs. {} from {} --> {}'.format(attack_method + "_transferred", args.transferred_attack_arch, plot_mi(args.save_path, attack_method + "_transferred")), log)

    return top1, val_loss

def ens_validate(val_loader, model, criterion, args, log, suffix=''):
    model.eval()
    # evaluation/adv attack step 1: enable parallel_eval
    if not args.full:
        parallel_eval(model)

    ece_func = _ECELoss().cuda(args.gpu)
    with torch.no_grad():
        targets = []
        mis = []
        preds = []
        top1, top5, val_loss = 0, 0, 0
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            targets.append(target)

            # evaluation/adv attack step 2: obtain features for estimating uncertainty (mi);
            # obtain output, which is in the shape [batch_size, num_mc_samples, num_classes]
            # and need to be transformed via outputs.softmax(-1).mean(-2) to get predicted probability
            if args.full:
                features, outputs = [], []
                for _ in range(args.num_mc_samples):
                    feature, output = model(input, return_features=True)
                    features.append(feature)
                    outputs.append(output)
                features = torch.stack(features, 1)
                outputs = torch.stack(outputs, 1)
            else:
                features, outputs = model(input, return_features=True)
            assert outputs.dim() == 3
            output = outputs.softmax(-1).mean(-2)
            if args.unc_metric == 'feature_var':
                mi = features.var(dim=1).mean(dim=[1,2,3])
            elif args.unc_metric == 'softmax_var':
                mi = outputs.softmax(-1).var(dim=1).mean(dim=[1])
            else:
                mi = (- output * output.log()).sum(1) - (- outputs.softmax(-1) * outputs.log_softmax(-1)).sum(2).mean(1)
            preds.append(output)
            mis.append(mi)
            loss = criterion((output+1e-10).log(), target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            top1 += prec1*target.size(0)
            top5 += prec5*target.size(0)
            val_loss += loss*target.size(0)

        preds = torch.cat(preds, 0)
        mis = torch.cat(mis, 0)
        targets = torch.cat(targets, 0)
        top1 /= targets.size(0)
        top5 /= targets.size(0)
        val_loss /= targets.size(0)

        # to sync
        confidences, predictions = torch.max(preds, 1)
        if args.distributed:
            if suffix == '':
                confidences = dist_collect(confidences)
                predictions = dist_collect(predictions)
                targets = dist_collect(targets)
            mis = dist_collect(mis)
            top1 = reduce_tensor(top1.data, args)
            top5 = reduce_tensor(top5.data, args)
            val_loss = reduce_tensor(val_loss.data, args)

        if suffix == '':
            ens_ece = ece_func(confidences, predictions, targets,
                os.path.join(args.save_path, 'ens_cal{}.pdf'.format(suffix)))
        else:
            ens_ece = None

    if args.gpu == 0:
        np.save(os.path.join(args.save_path, 'mis{}.npy'.format(suffix)),
            mis.data.cpu().numpy())

    # evaluation/adv attack step 3: disable parallel_eval
    if not args.full:
        disable_parallel_eval(model)
    return top1.item(), top5.item(), val_loss.item(), ens_ece.item() if ens_ece is not None else None

def ens_attack(val_loader, model, criterion, mean, std, stack_kernel, args, log, attack_method, attack_model=None):
    def _grad(X, y, mean, std):
        if args.evaluate and args.full and attack_model == model: # need an isolation evluation with small batch size to avoid OOM
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
                    features, outputs = attack_model(X.sub(mean).div(std), return_features=True)
                    outputs = outputs.softmax(-1)
                    if outputs.dim() == 3:
                        output = outputs.mean(-2) + 1e-10
                    else:
                        output = outputs
                    loss = F.cross_entropy(output.log(), y, reduction='none')
                    if args.mimicry != 0.:
                        loss -= features.var(dim=1).mean(dim=[1,2,3]) * args.mimicry
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
                features, outputs = attack_model(X_cw.sub(mean).div(std), return_features=True)
                outputs = outputs.softmax(-1)
                if outputs.dim() == 3:
                    logits = (outputs.mean(-2) + 1e-10).log()
                else:
                    logits = outputs.log()
                logit_target = torch.max(y_one_hot * logits, 1)[0]
                logit_other = torch.max(
                    (1 - y_one_hot) * logits - 1e6 * y_one_hot, 1)[0]
                loss = torch.mean(logit_other - logit_target)
                if args.mimicry != 0.:
                    loss -= features.var(dim=1).mean() * args.mimicry
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
    if not args.full:
        parallel_eval(model)
    if is_transferred:
        attack_model.eval()
        if not args.full:
            parallel_eval(attack_model)
    else:
        attack_model = model

    with torch.no_grad():
        losses, top1, top5, num_data = 0, 0, 0, 0
        mis = []
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True).mul_(std).add_(mean)
            target = target.cuda(args.gpu, non_blocking=True)

            X_adv = eval('_{}_whitebox'.format(attack_method))(input, target, mean, std)

            if args.full:
                features, outputs = [], []
                for _ in range(args.num_mc_samples):
                    feature, output = model(X_adv.sub(mean).div(std), return_features=True)
                    features.append(feature)
                    outputs.append(output)
                features = torch.stack(features, 1)
                outputs = torch.stack(outputs, 1)
            else:
                features, outputs = model(X_adv.sub(mean).div(std), return_features=True)

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
    if not args.full:
        disable_parallel_eval(model)
    if is_transferred and not args.full: disable_parallel_eval(attack_model)

def adjust_learning_rate_per_step(new_added_optimizer, epoch, i, num_ites_per_epoch, args):
    slr = args.ft_lr + (args.lr - args.ft_lr) * (1 + math.cos(math.pi * (epoch + float(i)/num_ites_per_epoch) / args.epochs)) / 2.
    for param_group in new_added_optimizer.param_groups: param_group['lr'] = slr
    return slr

if __name__ == '__main__': main()
