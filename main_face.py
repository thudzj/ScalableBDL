from __future__ import division
import os, sys, shutil, time, random, math
import argparse
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import torch
import torch.backends.cudnn as cudnn

import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import models.mobilenet as models
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, _ECELoss, load_dataset_face, plot_mi, plot_ens
from mean_field import *
from verification import verify

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Training script for face recognition', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('--data_path', metavar='DPATH', default='./data/faces_emore/', type=str, help='Path to dataset')
parser.add_argument('--arch', metavar='ARCH', default='mobilenet_v2', help='model architecture: ' + ' | '.join(model_names) + ' (default: mobilenet_v2)')

# Optimization
parser.add_argument('--epochs', metavar='N', type=int, default=90, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--ft_learning_rate', type=float, default=0.0001, help='The fine-tuning Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60, 80], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1, 0.1], help='LR is multiplied by gamma on schedule')

#Regularization
parser.add_argument('--decay', type=float, default=1e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--dropout_rate', type=float, default=0.) #0.2 for mc dropout

# Checkpoints
parser.add_argument('--save_path', type=str, default='./snapshots_ab_face/', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate model on test set')

# Acceleration
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')

# Random seed
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--job-id', type=str, default='')

# Bayesian
parser.add_argument('--bayes', type=str, default=None, help='Bayes type: None, mean field, matrix gaussian')
parser.add_argument('--fc_bayes', type=str, default=None)
parser.add_argument('--log_sigma_init_range', type=float, nargs='+', default=[-5, -4])
parser.add_argument('--log_sigma_lr', type=float, default=0.1)
parser.add_argument('--single_eps', action='store_true', default=False)
parser.add_argument('--local_reparam', action='store_true', default=False)

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

    log = open(os.path.join(args.save_path, 'log{}{}.txt'.format('_seed'+
                   str(args.manualSeed), '_eval' if args.evaluate else '')), 'w')
    log = (log, args.gpu)

    net = models.__dict__[args.arch](args, num_classes=10341)
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

    mus_pretrained, mus_new, vars = [], [], []
    for name, param in net.named_parameters():
        if 'log_sigma' in name:
            vars.append(param)
        else:
            assert(param.requires_grad)
            if 'classifier' in name:
                mus_new.append(param);
            else:
                mus_pretrained.append(param)

    optimizer = torch.optim.SGD([{'params': mus_pretrained, 'lr': args.ft_learning_rate},
                                 {'params': mus_new, 'lr': args.learning_rate}],
                                args.learning_rate, momentum=args.momentum,
                                weight_decay=args.decay)
    if args.bayes:
        assert(len(mus_pretrained) + len(mus_new) == len(vars))
        var_optimizer = VarSGD(vars, args.log_sigma_lr, num_data=None,
                               momentum=args.momentum, weight_decay=args.decay)
    else:
        assert(len(vars) == 0)
        var_optimizer = NoneOptimizer()

    pretrain_dict_imagenet = torch.load("ckpts/mobilenet_v2-b0353104.pth")
    model_dict = net.state_dict()
    pretrain_dict_imagenet = {'module.' + k: v for k, v in pretrain_dict_imagenet.items() if 'module.' + k in model_dict and model_dict['module.' + k].size() == v.size()}
    model_dict.update(pretrain_dict_imagenet)
    net.load_state_dict(model_dict)
    print("Initialized model with ImageNet pretrained weights")

    recorder = RecorderMeter(args.epochs)
    if args.resume:
        if args.resume == 'auto': args.resume = os.path.join(args.save_path, 'checkpoint.pth.tar')
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpu))
            if 'recorder' in checkpoint:
                recorder = checkpoint['recorder']
                recorder.refresh(args.epochs)
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            var_optimizer.load_state_dict(checkpoint['var_optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    cudnn.benchmark = True

    train_loader, test_loaders = load_dataset_face(args)
    var_optimizer.num_data = len(train_loader.dataset)

    if args.evaluate:
        validate(test_loaders, net, criterion, args, log)
        return

    start_time = time.time()
    epoch_time = AverageMeter()
    train_los = -1

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, var_optimizer, epoch, args)
        cur_ft_lr, cur_lr, cur_slr = optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], var_optimizer.param_groups[0]['lr'],

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f} {:6.4f} {:6.4f}]'.format(
                                    time_string(), epoch, args.epochs, need_time, cur_ft_lr, cur_lr, cur_slr) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        train_acc, train_los = train(train_loader, net, criterion, optimizer, var_optimizer, epoch, args, log)
        if epoch % 1 == 0:
            validate(test_loaders, net, criterion, args, log)
        recorder.update(epoch, train_los, train_acc, 0, 0)

        if args.gpu == 0:
            save_checkpoint({
              'epoch': epoch + 1,
              'arch': args.arch,
              'state_dict': net.state_dict(),
              'recorder': recorder,
              'optimizer' : optimizer.state_dict(),
              'var_optimizer' : var_optimizer.state_dict(),
            }, False, args.save_path, 'checkpoint.pth.tar')

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'log.png'))

    validate(test_loaders, net, criterion, args, log)

    log[0].close()

def train(train_loader, model, criterion, optimizer, var_optimizer, epoch, args, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(input)
        loss = criterion(output, target)

        losses.update(loss.item(), input.size(0))
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()
        var_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        var_optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    print_log('  Epoch: [{:03d}]   '
                'Time {batch_time.avg:.3f}   '
                'Data {data_time.avg:.3f}   '
                'Loss {loss_avg:.4f}   '
                'Prec@1 {top1.avg:.3f}   '
                'Prec@5 {top5.avg:.3f}   '.format(
                epoch, batch_time=batch_time, data_time=data_time, loss_avg=losses.avg,
                top1=top1, top5=top5) + time_string(), log)
    return top1.avg, losses.avg

def validate(val_loaders, model, criterion, args, log):
    model.eval()
    results = {}
    name, val_loader, issame = val_loaders[args.gpu % len(val_loaders)]
    with torch.no_grad():
        with model.no_sync():
            embeddings = []
            for i, input in enumerate(val_loader):
                input = input.cuda(args.gpu, non_blocking=True)
                output = model(input, True)
                norm = torch.norm(output, 2, 1, True)
                embedding = torch.div(output, norm)
                embeddings.append(embedding)
    embeddings = torch.cat(embeddings).data.cpu().numpy()
    tpr, fpr, accuracy, best_thresholds = verify(embeddings, issame, 10)
    results[name] = accuracy.mean()
    print_log('  **Test** {}'.format('  '.join(['{}: {:.3f}'.format(k, v) for k,v in results.items()])), log, True)
    torch.distributed.barrier()

def print_log(print_string, log, force=False):
    if log[1] == 0 or force:
        print("{}".format(print_string))
        log[0].write('{}\n'.format(print_string))
        log[0].flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, var_optimizer, epoch, args):
    lr = args.learning_rate
    slr = args.log_sigma_lr
    flr = args.ft_learning_rate
    assert len(args.gammas) == len(args.schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(args.gammas, args.schedule):
        if (epoch >= step): slr = slr * gamma; lr = lr * gamma; flr = flr * gamma
        else: break
    # lr = lr * np.prod(args.gammas)
    lr = max(lr, 5e-5)
    flr = max(flr, 5e-5)
    optimizer.param_groups[0]['lr'] = flr; optimizer.param_groups[1]['lr'] = lr
    for param_group in var_optimizer.param_groups: param_group['lr'] = slr
    return lr, slr

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

def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0)

def freeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)):
        m.deterministic = True

def unfreeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)):
        m.deterministic = False

if __name__ == '__main__': main()
