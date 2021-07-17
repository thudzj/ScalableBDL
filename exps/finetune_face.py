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
from face_verification import verify
from dataset.face import load_dataset_ft
import models.mobilenet as mobilenet
import models.model_irse as model_irse
import face_metrics
from models.utils import load_state_dict_from_url

# step 0: clone the ScalableBDL repo and checkout to the efficient branch
sys.path.insert(0, '../')
from scalablebdl.mean_field import to_bayesian as to_bayesian_mfg
from scalablebdl.empirical import to_bayesian as to_bayesian_emp
from scalablebdl.bnn_utils import freeze, unfreeze, set_mc_sample_id, \
    disable_dropout, parallel_eval, disable_parallel_eval

parser = argparse.ArgumentParser(description='Training script for Face', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', type=str,
                    default='/data/xiaoyang/data/faces_emore/')
parser.add_argument('--dataset', metavar='DSET', type=str, default='face')
parser.add_argument('--arch', metavar='ARCH', default='mobilenet_v2')
parser.add_argument('--head', type=str, default='Softmax')
# parser.add_argument('--transferred_attack_arch', metavar='ARCH', default='resnet152')

# Optimization
parser.add_argument('--epochs', metavar='N', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--ft_lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)

#Regularization
parser.add_argument('--decay', type=float, default=5e-4,
                    help='Weight decay')

# Checkpoints
parser.add_argument('--save_path', type=str, default='/data/zhijie/snapshots_ba/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--job-id', type=str, default='onelayer-face')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='Evaluate model on test set')

# Acceleration
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers (default: 4)')

# Random seed
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')

# Bayesian
parser.add_argument('--posterior_type', type=str, default='emp')
parser.add_argument('--psi_init_range', type=float, nargs='+', default=[-6, -5])
parser.add_argument('--alpha', type=float, default=1.)
parser.add_argument('--num_mc_samples', type=int, default=20)
parser.add_argument('--uncertainty_threshold', type=float, default=0.5)
parser.add_argument('--epsilon_max_train', type=float, default=2.)
parser.add_argument('--epsilon_min_train', type=float, default=1.)
parser.add_argument('--blur_prob', type=float, default=0.1)
parser.add_argument('--blur_kernel_size', type=int, nargs='+', default=[3, 5, 7, 9, 11])
parser.add_argument('--blur_sig', type=float, nargs='+', default=[0., 5.])
parser.add_argument('--mc_dropout', action='store_true', default=False)

# Attack settings
parser.add_argument('--attack_methods', type=str, nargs='+',
                    default=['FGSM', 'BIM', 'PGD', 'MIM', 'TIM', 'DI_MIM', 'FGSM_L2', 'BIM_L2', 'PGD_L2'])
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

class UnifiedModel(torch.nn.Module):
    def __init__(self, arch, net=None, backbone=None, head=None):
        super(UnifiedModel, self).__init__()
        self.arch = arch
        self.net = net
        self.backbone = backbone
        self.head = head

    def forward(self, input, target=None, return_features=False, is_logits=True):
        if self.arch == 'mobilenet_v2':
            return self.net(input, return_features, is_logits)
        elif self.arch == 'IR_50':
            x, y = self.backbone(input)
            if is_logits:
                if y.dim() == 3:
                    target_ = target[:, None].repeat(1, y.shape[1]).view(-1)
                    y = self.head(y.flatten(0, 1), target_).view(*y.shape[:2], -1)
                else:
                    y = self.head(y, target)

            if return_features:
                return x, y
            else:
                return y
        else:
            raise NotImplementedError

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
    args.use_all_face_data = (args.head == 'ArcFace')

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

    log = open(os.path.join(args.save_path, 'log_seed{}{}.txt'.format(
               args.manualSeed, '_eval' if args.evaluate else '')), 'w')
    log = (log, args.gpu)

    # step 1: convert the last res-block to be Bayesian
    # disable_dropout(net) # we may need to disable the dropout (not sure, need try)
    if args.arch == 'mobilenet_v2':
        net_ = mobilenet.__dict__[args.arch](pretrained=True, num_classes=10341)
        if args.posterior_type == 'emp':
            net_.features[-1] = to_bayesian_emp(net_.features[-1], args.num_mc_samples)
            net_.features[-2] = to_bayesian_emp(net_.features[-2], args.num_mc_samples)
        elif args.posterior_type == 'mfg':
            net_.features[-1] = to_bayesian_mfg(net_.features[-1], args.psi_init_range, args.num_mc_samples)
            net_.features[-2] = to_bayesian_mfg(net_.features[-2], args.psi_init_range, args.num_mc_samples)
        else:
            raise NotImplementedError
        net = UnifiedModel(args.arch, net=net_)
    elif args.arch == 'IR_50':
        backbone_ = model_irse.__dict__[args.arch]([112, 112])
        backbone_.load_state_dict(load_state_dict_from_url(
            'http://ml.cs.tsinghua.edu.cn/~zhijie/files/Backbone_IR_50_{}.pth'.format(args.head),
            map_location=lambda storage, loc: storage.cuda(args.gpu), progress=True))
        if args.mc_dropout:
            pass
        elif args.posterior_type == 'emp':
            backbone_.body[-1] = to_bayesian_emp(backbone_.body[-1], args.num_mc_samples)
            # backbone_.body[-1].shortcut_layer = to_bayesian_emp(torch.nn.Sequential(
            #         torch.nn.Conv2d(512, 512, (1, 1), 1, bias=False),
            #         torch.nn.BatchNorm2d(512),
            #     ), args.num_mc_samples, is_residual=True)
        elif args.posterior_type == 'mfg':
            backbone_.body[-1] = to_bayesian_mfg(backbone_.body[-1], args.psi_init_range, args.num_mc_samples)
            # backbone_.body[-1].shortcut_layer = to_bayesian_mfg(torch.nn.Sequential(
            #         torch.nn.Conv2d(512, 512, (1, 1), 1, bias=False),
            #         torch.nn.BatchNorm2d(512),
            #     ), args.psi_init_range, args.num_mc_samples, is_residual=True)
        else:
            raise NotImplementedError

        head_ = face_metrics.__dict__[args.head](in_features=512,
            out_features=10575 if args.use_all_face_data else 10341, device_id=None)
        head_.load_state_dict(load_state_dict_from_url(
            'http://ml.cs.tsinghua.edu.cn/~zhijie/files/Head_{}.pth'.format(args.head),
            map_location=lambda storage, loc: storage.cuda(args.gpu), progress=True))
        net = UnifiedModel(args.arch, backbone=backbone_, head=head_)
    else:
        raise NotImplementedError

    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch  version : {}".format(torch.__version__), log)
    print_log("CuDNN  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Number of parameters: {}".format(sum([p.numel() for p in net.parameters()])), log)
    print_log(str(args), log)

    # attack_net = models.__dict__[args.transferred_attack_arch](pretrained=True)

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
        # attack_net.cuda(args.gpu)
        # attack_net = torch.nn.parallel.DistributedDataParallel(attack_net, device_ids=[args.gpu])
    else:
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)
        # attack_net = attack_net.cuda(args.gpu)

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
    if not args.mc_dropout:
        if args.posterior_type == 'mfg':
            from scalablebdl.mean_field import PsiSGD
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
    train_loader, val_loaders = load_dataset_ft(args)

    mean = torch.from_numpy(np.array([0.5, 0.5, 0.5])).view(1,3,1,1).cuda(args.gpu).float()
    std = torch.from_numpy(np.array([0.5, 0.5, 0.5])).view(1,3,1,1).cuda(args.gpu).float()
    stack_kernel = gaussian_kernel().cuda(args.gpu)

    # step 2: set the num_data arg for new_added_optimizer if using mean-field Gaussian
    if not args.mc_dropout:
        if args.posterior_type == 'mfg':
            new_added_optimizer.num_data = len(train_loader.dataset)

    if args.evaluate or args.mc_dropout:
        evaluate(val_loaders, net,
                     criterion, mean, std, stack_kernel, args, log)
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
        # evaluate(val_loaders, net, criterion, mean, std, stack_kernel, args, log)
        recorder.update(epoch, train_los, train_acc, 0, 0)

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

    evaluate(val_loaders, net,
                 criterion, mean, std, stack_kernel, args, log)

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
            mc_sample_id = np.concatenate([np.random.randint(0, args.num_mc_samples, size=bs),
                np.stack([np.random.choice(args.num_mc_samples, 2, replace=False) for _ in range(bs1)]).T.reshape(-1)])
            set_mc_sample_id(model, args.num_mc_samples, mc_sample_id)

        # step 5: forward prop for predicted logits of normal data and features of noisy data
        target_ = torch.cat([target, torch.zeros(bs1*2, device=target.device)])
        features, output = model(torch.cat([input, input1.repeat(2, 1, 1, 1)]), target_, return_features=True)

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

def evaluate(val_loaders, net,
             criterion, mean, std, stack_kernel, args, log):

    ens_validate(val_loaders, net, criterion, args, log)

    for attack_method in args.attack_methods:
        ens_attack(val_loaders,
                   net, criterion, mean, std, stack_kernel, args, log, attack_method)
        if args.gpu == 0:
            for k in val_loaders:
                print_log('{} vs. {} --> {}'.format(k[0], attack_method, plot_mi(args.save_path, attack_method+'_'+k[0], k[0])), log)

    # for attack_method in args.attack_methods:
    #     if 'L2' in attack_method or 'FGSM' in attack_method or 'BIM' in attack_method:
    #         continue
    #     ens_attack(attack_loader,
    #                net, criterion, mean, std, stack_kernel, args, log, attack_method, attack_model=attack_net)
    #     if args.gpu == 0:
    #         print_log('NAT vs. {} from {} --> {}'.format(attack_method + "_transferred", args.transferred_attack_arch, plot_mi(args.save_path, attack_method + "_transferred")), log)

def ens_validate(val_loaders, model, criterion, args, log):
    model.eval()
    if args.mc_dropout:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'): m.train()
    # evaluation/adv attack step 1: enable parallel_eval
    parallel_eval(model)

    if isinstance(val_loaders, list):
        name, val_loader, issame = val_loaders[args.gpu % len(val_loaders)]
    else:
        name, val_loader, issame = val_loaders

    ece_func = _ECELoss().cuda(args.gpu)
    with torch.no_grad():
        with model.no_sync():
            embeddings = []
            mis = []
            for i, input in enumerate(val_loader):
                if isinstance(input, tuple) or isinstance(input, list): input = input[0]
                input = input.cuda(args.gpu, non_blocking=True)

                # evaluation/adv attack step 2: obtain features for estimating uncertainty (mi);
                # obtain output, which is in the shape [batch_size, num_mc_samples, num_features] and need to be transformed via outputs.mean(-2) to get predicted embedding
                if args.mc_dropout:
                    outputs = []
                    for _ in range(args.num_mc_samples):
                        output = model(input, is_logits=False)
                        outputs.append(output)
                    outputs = torch.stack(outputs, 1)
                    features = outputs
                else:
                    features, outputs = model(input, return_features=True, is_logits=False)
                assert outputs.dim() == 3

                embedding = outputs.mean(-2)
                norm = torch.norm(embedding, 2, 1, True)
                embeddings.append(torch.div(embedding, norm))

                mi = features.var(dim=1).mean(dim=[1,2,3] if not args.mc_dropout else [1])
                mis.append(mi)

    embeddings = torch.cat(embeddings).data.cpu().numpy()
    mis = torch.cat(mis, 0)

    if (isinstance(val_loaders, list) and args.gpu < len(val_loaders)) or \
                    ((not isinstance(val_loaders, list)) and args.gpu == 0):
        np.save(os.path.join(args.save_path, 'mis_{}.npy'.format(name)),
            mis.data.cpu().numpy())

    tpr, fpr, accuracy, best_thresholds = verify(embeddings, issame, 10)
    print_log('Parallel ensemble {} {}: {:.4f}'.format(args.num_mc_samples, name, accuracy.mean()), log, True)

    # evaluation/adv attack step 3: disable parallel_eval
    disable_parallel_eval(model)
    torch.distributed.barrier()

def ens_attack(val_loaders, model, criterion, mean, std, stack_kernel, args, log, attack_method, attack_model=None):
    def _grad(X, y, mean, std):
        with attack_model.no_sync():
            with torch.enable_grad():
                X.requires_grad_()
                if args.mc_dropout:
                    for m in attack_model.modules():
                        if m.__class__.__name__.startswith('Dropout'): m.eval()
                outputs = attack_model(X.sub(mean).div(std), is_logits=False)
                if args.mc_dropout:
                    for m in attack_model.modules():
                        if m.__class__.__name__.startswith('Dropout'): m.train()
                if outputs.dim() == 3:
                    output = outputs.mean(-2).reshape(X.size(0)//2, 2, outputs.size(-1))
                else:
                    output = outputs.reshape(X.size(0)//2, 2, outputs.size(-1))

                loss = ((output[:, 0, :] - y[:, 1, :].detach())**2).sum(1) \
                     + ((output[:, 1, :] - y[:, 0, :].detach())**2).sum(1)
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
    if args.mc_dropout:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'): m.train()
    parallel_eval(model)
    if is_transferred:
        assert False
    else:
        attack_model = model

    name, val_loader, issame = val_loaders[args.gpu % len(val_loaders)]

    with torch.no_grad():
        with model.no_sync():
            mis = []
            embeddings = []
            for i, input in enumerate(val_loader):
                is_pair = issame[i*args.batch_size//2:min(len(issame), i*args.batch_size//2+args.batch_size//2)]
                if np.all(is_pair == False): continue

                input = input.cuda(args.gpu, non_blocking=True).mul_(std).add_(mean)
                input = input.reshape(args.batch_size//2, 2, 3, 112, 112)
                assert len(is_pair) == input.shape[0], (len(is_pair), input.shape[0])

                mask = torch.from_numpy(is_pair).cuda(args.gpu, non_blocking=True) == True
                input = input[mask, :, :, :, :].view(-1, 3, 112, 112)

                if args.mc_dropout:
                    for m in model.modules():
                        if m.__class__.__name__.startswith('Dropout'): m.eval()
                target = model(input.sub(mean).div(std), is_logits=False)
                if args.mc_dropout:
                    for m in model.modules():
                        if m.__class__.__name__.startswith('Dropout'): m.train()
                if target.dim() == 3:
                    target = target.mean(-2).reshape(input.size(0)//2, 2, -1)
                else:
                    target = target.reshape(input.size(0)//2, 2, -1)
                X_adv = eval('_{}_whitebox'.format(attack_method))(input, target, mean, std)

                if args.mc_dropout:
                    outputs = []
                    for _ in range(args.num_mc_samples):
                        output = model(X_adv.sub(mean).div(std), is_logits=False)
                        outputs.append(output)
                    outputs = torch.stack(outputs, 1)
                    features = outputs
                else:
                    features, outputs = model(X_adv.sub(mean).div(std), return_features=True, is_logits=False)

                # embedding = outputs.mean(-2)
                # norm = torch.norm(embedding, 2, 1, True)
                # embeddings.append(torch.div(embedding, norm))

                mi = features.var(dim=1).mean(dim=[1,2,3] if not args.mc_dropout else [1])
                mis.append(mi)

    mis = torch.cat(mis, 0)

    if args.gpu < len(val_loaders):
        np.save(os.path.join(args.save_path, 'mis_{}_{}{}.npy'.format(attack_method, name, "_transferred" if is_transferred else "")), mis.data.cpu().numpy())

    disable_parallel_eval(model)
    torch.distributed.barrier()


def adjust_learning_rate_per_step(new_added_optimizer, epoch, i, num_ites_per_epoch, args):
    slr = args.ft_lr + (args.lr - args.ft_lr) * (1 + math.cos(math.pi * (epoch + float(i)/num_ites_per_epoch) / args.epochs)) / 2.
    for param_group in new_added_optimizer.param_groups: param_group['lr'] = slr
    return slr

if __name__ == '__main__': main()
