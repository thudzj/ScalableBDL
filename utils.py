import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import os, sys, time
import numpy as np
import scipy.io as sio
import pandas as pd
import json
import bcolz

import skimage.transform
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})

import seaborn as sns
from sklearn import metrics
from augmentations import RandAugment
from random import random, choice
from scipy.ndimage.filters import gaussian_filter
import cv2

from io import BytesIO
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def ent(p):
    return -(p*p.log()).sum(-1)

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def refresh(self, epochs):
    if epochs == self.total_epoch: return
    self.epoch_losses = np.vstack( (self.epoch_losses, np.zeros((epochs - self.total_epoch, 2), dtype=np.float32) - 1) )
    self.epoch_accuracy = np.vstack( (self.epoch_accuracy, np.zeros((epochs - self.total_epoch, 2), dtype=np.float32)) )
    self.total_epoch = epochs

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == val_acc

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()

  def plot_curve(self, save_path):
    title = 'the accuracy/loss curve of train/val'
    dpi = 80
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)

    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)


    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
      # print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)

class _ECELoss(torch.nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

        bin_boundaries_plot = torch.linspace(0, 1, 11)
        self.bin_lowers_plot = bin_boundaries_plot[:-1]
        self.bin_uppers_plot = bin_boundaries_plot[1:]

    def forward(self, confidences, predictions, labels, title):
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        accuracy_in_bin_list = []
        for bin_lower, bin_upper in zip(self.bin_lowers_plot, self.bin_uppers_plot):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            accuracy_in_bin = 0
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean().item()
            accuracy_in_bin_list.append(accuracy_in_bin)

        fig = plt.figure(figsize=(4,3))
        p1 = plt.bar(np.arange(10) / 10., accuracy_in_bin_list, 0.1, align = 'edge', edgecolor ='black')
        p2 = plt.plot([0,1], [0,1], '--', color='gray')

        plt.ylabel('Accuracy', fontsize=14)
        plt.xlabel('Confidence', fontsize=14)
        #plt.title(title)
        plt.xticks(np.arange(0, 1.01, 0.2))
        plt.yticks(np.arange(0, 1.01, 0.2))
        plt.xlim(left=0,right=1)
        plt.ylim(bottom=0,top=1)
        plt.grid(True)
        #plt.legend((p1[0], p2[0]), ('Men', 'Women'))
        plt.text(0.1, 0.83, 'ECE: {:.4f}'.format(ece.item()), fontsize=14)
        plt.savefig(title, format='pdf', dpi=600, bbox_inches='tight')
        return ece

class GeneratedDataAugment(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, img):
        img = np.array(img)

        if random() < self.args.blur_prob:
            sig = sample_continuous(self.args.blur_sig)
            gaussian_blur(img, sig)

        if random() < self.args.jpg_prob:
            method = sample_discrete(self.args.jpg_method)
            qual = sample_discrete(list(range(self.args.jpg_qual[0], self.args.jpg_qual[1] + 1)))
            img = jpeg_from_key(img, qual, method)
        return Image.fromarray(img)

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]

def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

def jpeg_from_key(img, compress_val, key):
    if key == 'cv2':
        return cv2_jpg(img, compress_val)
    else:
        return pil_jpg(img, compress_val)

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

def time_file_str():
  ISOTIMEFORMAT='%Y-%m-%d'
  string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string + '-{}'.format(random.randint(1, 10000))

def plot_mi(dir_, type_, type2_=None):
    if type2_ is None:
        mi_nat = np.load(os.path.join(dir_, 'mis.npy'))
        label2_ = 'Normal'
    else:
        mi_nat = np.load(os.path.join(dir_, 'mis_{}.npy'.format(type2_)))
        label2_ = type2_#.capitalize()
    mi_svhn = np.load(os.path.join(dir_, 'mis_{}.npy'.format(type_)))
    fig = plt.figure()

    if type_ == 'advg':
        label_ = 'Adversarial'
    elif type_ == 'adv':
        label_ = 'adversarial-T'
    elif 'adv_' in type_:
        label_ = 'Adversarial'
    elif type_ == "svhn":
        label_ = 'SVHN'
    elif type_ == "celeba":
        label_ = 'CelebA'
    elif type_ == "noise":
        label_ = 'noise'
    elif type_ == "fake":
        label_ = 'Fake'
    elif type_ == "fake2":
        label_ = 'Fake'
    else:
        raise NotImplementedError

    # Draw the density plot
    sns.distplot(mi_nat, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'clip': (-0.0001, 3)},
                 label = label2_)
    sns.distplot(mi_svhn, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'clip': (-0.0001, 3)},
                 label = label_)

    x = np.concatenate((mi_nat, mi_svhn), 0)
    y = np.zeros(x.shape[0])
    y[mi_nat.shape[0]:] = 1

    ap = metrics.average_precision_score(y, x)

    # Plot formatting
    plt.legend()#(prop={'size': 20})
    plt.xlabel('Mutual information')#, fontsize=20)
    plt.ylabel('Density')#, fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_, '{}_vs_{}.pdf'.format('nat' if type2_ is None else type2_, type_)), bbox_inches='tight')
    return ap

def plot_ens(dir_, rets, baseline_acc):
    lw = 1.25
    color = ['red', 'green', 'darkorange', 'b']
    if isinstance(rets, list):
        rets = np.stack([np.array(item) for item in rets])
    min_acc = min(rets[:, 2].min(), rets[:, 6].min(), baseline_acc) - 0.1
    max_acc = max(rets[:, 2].max(), rets[:, 6].max(), baseline_acc) + 0.1

    fig = plt.figure(figsize=(4,3))
    fig, ax1 = plt.subplots(figsize=(4,3))
    l1 = ax1.plot(rets[:, 0]+1, rets[:, 2], color=color[0], lw=lw, alpha=0.6)
    l2 = ax1.plot(rets[:, 0]+1, rets[:, 6], color=color[1], lw=lw)
    l3 = ax1.plot(rets[:, 0]+1, np.ones(rets.shape[0])*baseline_acc, color=color[2], lw=lw, alpha=0.6, linestyle='dashed')
    ax1.set_yticks(np.arange(1, 101, 1))
    ax1.set_xticks([1,] + list(np.arange(20, rets.shape[0]+1, 20)))
    ax1.set_ylim((min_acc, max_acc))
    ax1.set_xlim((1, rets.shape[0]))
    ax1.set_xlabel('The number of MC sample')
    ax1.set_ylabel('Test accuracy (%)')
    ax1.legend(l1+l2+l3, ['Individual', 'Ensemble', 'Deterministic'], loc = 'best', fancybox=True, columnspacing=0.5, handletextpad=0.2, borderpad=0.15) # +l3+l4 , 'Indiv ECE', 'Ensemble ECE'  , fontsize=11
    plt.savefig(os.path.join(dir_, 'ens_plot.pdf'), format='pdf', dpi=600, bbox_inches='tight')


#------------------------------------------------------Cifar------------------------------------------------------#

def load_dataset(args):
    if args.dataset == 'cifar10':
        mean, std = [x / 255 for x in [125.3, 123.0, 113.9]],  [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = dset.CIFAR10
    elif args.dataset == 'cifar100':
        mean, std = [x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]]
        dataset = dset.CIFAR100

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)])
        if args.cutout: train_transform.transforms.append(Cutout(n_holes=1, length=16))
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        train_data = dataset(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dataset(args.data_path, train=False, transform=test_transform, download=True)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data) if args.distributed else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    else: assert False, 'Do not support dataset : {}'.format(args.dataset)

    return train_loader, test_loader

class CIFARR:
    def __init__(self, args, eps=0.031, transform=None, transform_last=None):
        super(CIFARR, self).__init__()
        self.transform_last = transform_last
        self.eps = eps
        self.max_choice = args.max_choice if args.max_choice else 2
        self.num_gan = args.num_gan
        self.cutout = Cutout(n_holes=1, length=16) if args.cutout else None
        if args.dataset == 'cifar10': dataset = dset.CIFAR10
        elif args.dataset == 'cifar100': dataset = dset.CIFAR100
        else: raise NotImplementedError
        self.normal_data = dataset(args.data_path, train=True, transform=transform, download=True)

        self.fake_data = dset.ImageFolder("gan_samples/cifar10/train",
            transforms.Compose([
                GeneratedDataAugment(args),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]))
        rng = np.random.RandomState(0)
        rand_idx = rng.permutation(len(self.fake_data.samples))[:self.num_gan]
        self.fake_data.samples = [self.fake_data.samples[i] for i in rand_idx]
        self.fake_data.targets = None

    def __getitem__(self, index):
        choice = np.random.choice(self.max_choice)
        if choice == 0: # add l_{inf} noise
            img, _ = self.normal_data[index]
            img = torch.empty_like(img).uniform_(-self.eps, self.eps).add_(img).clamp_(0, 1)
        elif choice == 1: # fake
            rand_idx = np.random.randint(self.num_gan)
            img, _ = self.fake_data[rand_idx]
        else:
            raise NotImplementedError

        img = self.transform_last(img)
        if self.cutout: img = self.cutout(img)
        return img

    def __len__(self):
        return len(self.normal_data)

def load_dataset_ft(args):
    if args.dataset == 'cifar10':
        mean, std = [x / 255 for x in [125.3, 123.0, 113.9]],  [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = dset.CIFAR10
    elif args.dataset == 'cifar100':
        mean, std = [x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]]
        dataset = dset.CIFAR100

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)])
        if args.cutout: train_transform.transforms.append(Cutout(n_holes=1, length=16))
        train_data = dataset(args.data_path, train=True, transform=train_transform, download=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size//2, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        train_data1 = CIFARR(args, args.epsilon*args.epsilon_scale, transform_last=transforms.Normalize(mean, std),
                            transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4), transforms.ToTensor()]))
        train_sampler1 = torch.utils.data.distributed.DistributedSampler(train_data1) if args.distributed else None
        train_loader1 = torch.utils.data.DataLoader(train_data1, batch_size=args.batch_size//4, shuffle=(train_sampler1 is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler1)

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

    return train_loader, train_loader1, test_loader, test_loader1, fake_loader, fake_loader2 #celeba_loader

#----------------------------------------------------ImageNet----------------------------------------------------#

def load_dataset_in(args):
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = dset.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader


class ImageNetR:
    def __init__(self, args, eps=16./255., transform=None, transform_last=None):
        super(ImageNetR, self).__init__()
        self.transform_last = transform_last
        self.eps = eps
        self.max_choice = args.max_choice if args.max_choice else 2
        self.num_gan = args.num_gan

        self.normal_data = dset.ImageFolder(os.path.join(args.data_path, 'train'), transform)
        self.fake_data = dset.ImageFolder("gan_samples/imagenet/train",
            transforms.Compose([
                GeneratedDataAugment(args),
                transforms.RandomCrop(224),
                #RandAugment(args.aug_n, args.aug_m),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]))
        rng = np.random.RandomState(0)
        rand_idx = rng.permutation(len(self.fake_data.samples))[:self.num_gan]
        self.fake_data.samples = [self.fake_data.samples[i] for i in rand_idx]
        self.fake_data.targets = None

    def __getitem__(self, index):
        choice = np.random.choice(self.max_choice)
        if choice == 0: # add l_{inf} noise
            img, _ = self.normal_data[index]
            img = torch.empty_like(img).uniform_(-self.eps, self.eps).add_(img).clamp_(0, 1)
        elif choice == 1: # fake
            rand_idx = np.random.randint(self.num_gan)
            img, _ = self.fake_data[rand_idx]
        else:
            raise NotImplementedError
        img = self.transform_last(img)
        return img

    def __len__(self):
        return len(self.normal_data)

def load_dataset_in_ft(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = dset.ImageFolder(
        os.path.join(args.data_path, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size//2, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_dataset1 = ImageNetR(args, args.epsilon*args.epsilon_scale, transform_last=normalize,
                              transform=transforms.Compose([
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()
                                        ]))
    train_sampler1 = torch.utils.data.distributed.DistributedSampler(train_dataset1) if args.distributed else None
    train_loader1 = torch.utils.data.DataLoader(
        train_dataset1, batch_size=args.batch_size//4, shuffle=(train_sampler1 is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler1)

    val_dataset = dset.ImageFolder(
        os.path.join(args.data_path, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    val_sampler1 = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    val_loader1 = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size//2, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler1)

    fake_dataset = dset.ImageFolder(
        "gan_samples/imagenet/val",
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    fake_sampler = torch.utils.data.distributed.DistributedSampler(fake_dataset) if args.distributed else None
    fake_loader = torch.utils.data.DataLoader(
        fake_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=fake_sampler)

    adv_dataset = dset.ImageFolder(
        "adv_samples/fgsm_resnet_152",
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    adv_sampler = torch.utils.data.distributed.DistributedSampler(adv_dataset) if args.distributed else None
    adv_loader = torch.utils.data.DataLoader(
        adv_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=adv_sampler)

    return train_loader, train_loader1, val_loader, val_loader1, fake_loader, adv_loader

def fast_collate(batch, memory_format=None):

    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
            memory_format=torch.contiguous_format if memory_format is None else memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets

class data_prefetcher():
    def __init__(self, loader, device=None):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=self.device)
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream(device=self.device))
        if target is not None:
            target.record_stream(torch.cuda.current_stream(device=self.device))
        self.preload()
        return input, target

#----------------------------------------------------face----------------------------------------------------#

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

class TensorsDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, transforms=None):
        self.data_tensor = data_tensor
        if transforms is None: transforms = []
        if not isinstance(transforms, list): transforms = [transforms]
        self.transforms = transforms
    def __getitem__(self, index):
        data_tensor = self.data_tensor[index]
        for transform in self.transforms:
            data_tensor = transform(data_tensor)
            return data_tensor
    def __len__(self):
        return self.data_tensor.size(0)

def load_dataset_face(args, INPUT_SIZE=[112, 112],
                      RGB_MEAN = [0.5, 0.5, 0.5], RGB_STD = [0.5, 0.5, 0.5],
                      val_datasets=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30', 'calfw', 'cplfw', 'vgg2_fp']):
    train_transform = transforms.Compose([
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN, std = RGB_STD),
    ])
    train_data = dset.ImageFolder(os.path.join(args.data_path, 'CASIA-maxpy-align'), train_transform)
    weights = torch.DoubleTensor(make_weights_for_balanced_classes(train_data.imgs, len(train_data.classes)))
    if args.distributed:
        from catalyst.data.sampler import DistributedSamplerWrapper
        train_sampler = DistributedSamplerWrapper(
            torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)))
    else:
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size = args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler = train_sampler
    )

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),
        transforms.CenterCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN, std = RGB_STD)
    ])
    val_loaders = []
    for name in val_datasets:
        carray = bcolz.carray(rootdir = os.path.join(args.data_path, name), mode = 'r')
        val_data_tensor = torch.tensor(carray[:, [2, 1, 0], :, :]) * 0.5 + 0.5
        val_data = TensorsDataset(val_data_tensor, val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size = args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=None)
        issame = np.load('{}/{}_list.npy'.format(args.data_path, name))
        val_loaders.append((name, val_loader, issame))

    return train_loader, val_loaders

class FaceR:
    def __init__(self, args, eps=16./255., transform=None, transform_last=None):
        super(FaceR, self).__init__()
        self.transform_last = transform_last
        self.eps = eps
        self.max_choice = args.max_choice if args.max_choice else 2
        self.num_gan = args.num_gan

        self.normal_data = dset.ImageFolder(os.path.join(args.data_path, 'CASIA-maxpy-align'), transform)
        self.fake_data = dset.ImageFolder("deepfake_samples/face/train_3",
            transforms.Compose([
                GeneratedDataAugment(args),
                transforms.Resize([128, 128]),
                transforms.RandomCrop([112, 112]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]))
        rng = np.random.RandomState(0)
        rand_idx = rng.permutation(len(self.fake_data.samples))[:self.num_gan]
        self.fake_data.samples = [self.fake_data.samples[i] for i in rand_idx]
        self.fake_data.targets = None

    def __getitem__(self, index):
        choice = np.random.choice(self.max_choice)
        if choice == 0: # add l_{inf} noise
            img, _ = self.normal_data[index]
            img = torch.empty_like(img).uniform_(-self.eps, self.eps).add_(img).clamp_(0, 1)
        elif choice == 1: # fake
            rand_idx = np.random.randint(self.num_gan)
            img, _ = self.fake_data[rand_idx]
        else:
            raise NotImplementedError
        img = self.transform_last(img)
        return img

    def __len__(self):
        return len(self.normal_data)

def load_dataset_face_ft(args, INPUT_SIZE=[112, 112],
                      RGB_MEAN = [0.5, 0.5, 0.5], RGB_STD = [0.5, 0.5, 0.5],
                      val_datasets=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30', 'calfw', 'cplfw', 'vgg2_fp']):
    normalize = transforms.Normalize(mean = RGB_MEAN, std = RGB_STD)
    train_transform = transforms.Compose([
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_data = dset.ImageFolder(os.path.join(args.data_path, 'CASIA-maxpy-align'), train_transform)
    weights = torch.DoubleTensor(make_weights_for_balanced_classes(train_data.imgs, len(train_data.classes)))
    if args.distributed:
        from catalyst.data.sampler import DistributedSamplerWrapper
        train_sampler = DistributedSamplerWrapper(
            torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)))
    else:
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size = args.batch_size//2, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler = train_sampler
    )

    train_data1 = FaceR(args, args.epsilon*args.epsilon_scale, transform_last=normalize,
                              transform=transforms.Compose([
                                            transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
                                            transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()
                                        ]))
    if args.distributed:
        train_sampler1 = DistributedSamplerWrapper(
            torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)))
    else:
        train_sampler1 = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader1 = torch.utils.data.DataLoader(
        train_data1, batch_size=args.batch_size//4, shuffle=(train_sampler1 is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler1)

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),
        transforms.CenterCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.ToTensor(),
        normalize
    ])
    val_loaders = []
    for name in val_datasets:
        carray = bcolz.carray(rootdir = os.path.join(args.data_path, name), mode = 'r')
        val_data_tensor = torch.tensor(carray[:, [2, 1, 0], :, :]) * 0.5 + 0.5
        val_data = TensorsDataset(val_data_tensor, val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size = args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=None)
        issame = np.load('{}/{}_list.npy'.format(args.data_path, name))
        val_loaders.append((name, val_loader, issame))

    fake_data = dset.ImageFolder(
        "deepfake_samples/face/val_3",
        transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            normalize,
        ]))
    fake_loader = torch.utils.data.DataLoader(
        fake_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)

    return train_loader, train_loader1, val_loaders, fake_loader

