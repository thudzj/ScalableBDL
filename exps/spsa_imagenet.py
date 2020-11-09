from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms

import torchvision

import models.resnet as models
from models.resnet import Bottleneck
from models.utils import load_state_dict_from_url
from utils import plot_mi, dist_collect

import os
import argparse
import sys
import datetime
import random
import numpy as np

sys.path.insert(0, '../')
from scalablebdl.empirical import to_bayesian as to_bayesian_emp
from scalablebdl.bnn_utils import parallel_eval, disable_parallel_eval


parser = argparse.ArgumentParser()
parser.add_argument('--save-path', type=str, default=None, help='save path')
parser.add_argument('--posterior_type', type=str, default='emp')
parser.add_argument('--psi_init_range', type=float, nargs='+', default=[-6, -5])
parser.add_argument('--num_mc_samples', type=int, default=20)
parser.add_argument('--full', action='store_true', default=False)
parser.add_argument('--mc_dropout', action='store_true', default=False)

# dataset dependent
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--samples_per_draw', default=128, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--sigma', default=1e-3, type=float)
parser.add_argument('--iterations', default=100, type=int)
parser.add_argument('--epsilon', default=16.0/255.0, type=float)
parser.add_argument('--num_test', default=5000, type=int)

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


def main():
    args = parser.parse_args()
    args.num_classes = 1000
    args.image_size = 224

    args.use_cuda = torch.cuda.is_available()
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

    if args.mc_dropout:
        net = models.__dict__['resnet50'](pretrained=False)
        for module in net.modules():
            if isinstance(module, Bottleneck):
                setattr(module, 'conv2', torch.nn.Sequential(module.conv2, torch.nn.Dropout(p=0.2)))
                setattr(module, 'conv3', torch.nn.Sequential(module.conv3, torch.nn.Dropout(p=0.2)))
        net.load_state_dict(load_state_dict_from_url(
            'http://ml.cs.tsinghua.edu.cn/~zhijie/files/resnet50-dropout0.2-afterbn.pth',
            progress=True))
        args.full = True
    else:
        net = models.__dict__['resnet50'](pretrained=True)
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
        elif args.posterior_type == 'emp':
            net.layer4[-1].conv1 = to_bayesian_emp(net.layer4[-1].conv1, args.num_mc_samples)
            net.layer4[-1].bn1 = to_bayesian_emp(net.layer4[-1].bn1, args.num_mc_samples)
            net.layer4[-1].conv2 = to_bayesian_emp(net.layer4[-1].conv2, args.num_mc_samples)
            net.layer4[-1].bn2 = to_bayesian_emp(net.layer4[-1].bn2, args.num_mc_samples)
            net.layer4[-1].conv3 = to_bayesian_emp(net.layer4[-1].conv3, args.num_mc_samples)
            net.layer4[-1].bn3 = to_bayesian_emp(net.layer4[-1].bn3, args.num_mc_samples)
        else:
            raise NotImplementedError

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url+":"+args.dist_port,
                                world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.gpu)
        net.cuda(args.gpu)
    else:
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)

    if args.save_path:
        if not os.path.isdir(args.save_path): os.makedirs(args.save_path)
        if os.path.isfile(os.path.join(args.save_path, 'checkpoint.pth.tar')):
            checkpoint = torch.load(os.path.join(args.save_path, 'checkpoint.pth.tar'), map_location='cuda:{}'.format(args.gpu))
            net.load_state_dict({k.replace("module.", ""): v for k,v in checkpoint['state_dict'].items()})

    # Data
    mean = torch.from_numpy(np.array(
        [0.485, 0.456, 0.406])).view(1,3,1,1).float().cuda(args.gpu)
    std = torch.from_numpy(np.array(
        [0.229, 0.224, 0.225])).view(1,3,1,1).float().cuda(args.gpu)

    testset = torchvision.datasets.ImageFolder(
        '/data/LargeData/Large/ImageNet/val',
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]))
    testsampler = DistributedSampler(testset, shuffle=False) if args.distributed else None
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=1, sampler=testsampler)

    cudnn.benchmark = True
    net.eval()
    if args.mc_dropout:
        for m in net.modules():
            if m.__class__.__name__.startswith('Dropout'): m.train()
    if not args.full:
        parallel_eval(net)

    per_image_acc = np.zeros([50000])
    queries = np.zeros([50000])
    mis = []
    norms = []
    for idx, (data, target) in enumerate(testloader):
        if args.gpu == 0 and idx < 20: print('test sample: ', idx)
        X, y = data.cuda(args.gpu).squeeze(0), target.cuda(args.gpu)
        per_image_acc[idx], queries[idx], mi, norm_ = spsa_blackbox(net, X, y, args, mean, std, logging=False)
        mis.append(mi)
        norms.append(norm_)
        if idx % 50 == 49:
            torch.distributed.barrier()
            mis_ = torch.cat(mis, 0)
            norms_ = torch.stack(norms)
            if args.distributed:
                mis_ = dist_collect(mis_)
                norms_ = dist_collect(norms_)

            mis_ = mis_[mis_ != -1]
            norms_ = norms_[norms_ != -1]

            if args.gpu == 0:
                print(idx, np.mean(per_image_acc[:idx+1]), np.mean(queries[:idx+1]), mis_.mean().item(), norms_.mean().item(), mis_.shape)
                np.save(os.path.join(args.save_path, 'mis_spsa.npy'), mis_.data.cpu().numpy())
                print('NAT vs. {} --> {}'.format('spsa', plot_mi(args.save_path, 'spsa')))

            if idx >= args.num_test/float(args.world_size):
                break
    if not args.full:
        disable_parallel_eval(net)
    # mis = torch.cat(mis, 0)
    # if args.distributed: mis = dist_collect(mis)
    #
    # print(np.sum(per_image_acc), np.mean(queries), np.mean(mis))
    # np.save(os.path.join(args.save_path.replace("checkpoint.pth.tar", ''), 'mis_spsa.npy'), mis.data.cpu().numpy())
    # if args.gpu == 0: print('NAT vs. {} --> {}'.format('spsa', plot_mi(args.save_path.replace("checkpoint.pth.tar", ''), 'spsa')))


class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = F.one_hot(targets, num_classes=self.num_classes)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1e6, dim=1)[0]

        loss = -(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            loss = torch.mean(loss)

        return loss

def spsa_blackbox(model,
                  xs,
                  ys,
                  args,
                  mean,
                  std,
                  logging=True):
      batch_size=args.batch_size
      samples_per_draw=args.samples_per_draw
      lr=args.learning_rate
      sigma=args.sigma
      iterations=args.iterations
      epsilon=args.epsilon
      num_classes=args.num_classes

      assert(len(xs.shape) == 3)
      assert((samples_per_draw // 2) % batch_size == 0)

      with torch.no_grad():
          if args.full:
              tmp = model(xs.unsqueeze(0).sub(mean).div(std).repeat(args.num_mc_samples, 1, 1, 1)).unsqueeze(0)
              pred = tmp.softmax(-1).mean(-2).max(1)[1]
              # outputs = []
              # for _ in range(args.num_mc_samples):
              #     output =
              #     outputs.append(output)
              # outputs = torch.stack(outputs, 1)
              # pred = outputs.softmax(-1).mean(-2).max(1)[1]
              # if args.gpu ==0 : print(tmp.var(1).mean())
          else:
              pred = model(xs.unsqueeze(0).sub(mean).div(std)).softmax(-1).mean(-2).max(1)[1]
      if pred != ys:
        return False, 0, torch.tensor([-1.], device=xs.device), torch.tensor(-1., device=xs.device)

      xs_adv = Variable(xs.data.clone(), requires_grad=True)
      ys_repeat = ys.repeat(batch_size * 2)
      opt = torch.optim.Adam([xs_adv], lr=lr)
      for _ in range(1, iterations + 1):
          with torch.no_grad():
              grad = 0
              for i in range(0, samples_per_draw // 2, batch_size):
                  pert = (torch.rand([batch_size] + list(xs_adv.shape), device=xs.device) * 2 - 1).sign()
                  pert = torch.cat([pert, -pert], 0)
                  eval_points = (xs_adv + sigma * pert).sub_(mean).div_(std)

                  if args.full:
                      # outputs = []
                      # for __ in range(args.num_mc_samples):
                      #     output =
                      #     outputs.append(output)
                      outputs = torch.stack([model(eval_points) for __ in range(args.num_mc_samples)], 1)
                      # if args.gpu ==0 : print(outputs.var(1).mean())
                  else:
                      outputs = model(eval_points)

                  losses = CWLoss(num_classes, reduce=False)(outputs.softmax(-1).mean(-2).add(1e-10).log(), ys_repeat)
                  grad -= (losses.view([batch_size * 2, 1, 1, 1]) * pert).mean(0)
              grad = grad / sigma / ((samples_per_draw // 2) / batch_size)

          opt.zero_grad()
          xs_adv.backward(grad)
          opt.step()

          with torch.no_grad():
            xs_adv.data = torch.min(torch.max(xs_adv.data, xs.data - epsilon), xs.data + epsilon)
            xs_adv.data.clamp_(0.0, 1.0)

            if args.full:
                features, outputs = model(xs_adv.unsqueeze(0).sub(mean).div(std).repeat(args.num_mc_samples, 1, 1, 1), return_features=True)
                features = features.unsqueeze(0)
                outputs = outputs.unsqueeze(0)
            else:
                features, outputs = model(xs_adv.unsqueeze(0).sub(mean).div(std), return_features=True)

            xs_adv_label = outputs.softmax(-1).mean(-2).max(1)[1]
            mi = features.var(dim=1).mean(dim=[1,2,3])
            norm_ = torch.max(torch.abs(xs_adv - xs))
          # print(_, xs_adv_label, ys, norm_)

          if logging:
              print("iteration:{}, loss:{}, learning rate:{}, "
                    "prediction:{}, distortion:{}, mi:{}".format(
                  _, losses.mean(), lr, xs_adv_label.item(),
                  norm_, mi.item()
              ))

          if xs_adv_label != ys:
              del opt
              del xs_adv
              return False, _, mi, norm_

      del opt
      del xs_adv
      return True, iterations + 1, mi, norm_

if __name__ == '__main__': main()
