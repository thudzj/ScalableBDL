import os
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from .utils import GeneratedDataAugment

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

def load_dataset(args):
    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = dset.CIFAR10
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        dataset = dset.CIFAR100

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=1, length=16))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_data = dataset(args.data_path, train=True,
                             transform=train_transform, download=True)
        test_data = dataset(args.data_path, train=False,
                            transform=test_transform, download=True)

        train_sampler = DistributedSampler(train_data) if args.distributed else None
        test_sampler = DistributedSampler(test_data) if args.distributed else None

        train_loader = torch.utils.data.DataLoader(train_data,
            batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(test_data,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    return train_loader, test_loader

class OODData:
    def __init__(self, args, transform_last=None):
        super(OODData, self).__init__()
        self.adv_eps = args.epsilon
        self.transform_last = transform_last
        self.num_fake = args.num_fake
        self.cutout = Cutout(n_holes=1, length=16) if args.cutout else None
        if args.dataset == 'cifar10':
            dataset = dset.CIFAR10
        elif args.dataset == 'cifar100':
            dataset = dset.CIFAR100
        else:
            raise NotImplementedError
        self.normal_data = dataset(args.data_path, train=True, download=True,
            transform=transforms.Compose([
               transforms.RandomCrop(32, padding=4),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor()
            ]))
        self.fake_data = dset.ImageFolder(os.path.join(args.data_path_fake, 'train'),
            transforms.Compose([
                GeneratedDataAugment(args),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]))
        rng = np.random.RandomState(0)
        rand_idx = rng.permutation(len(self.fake_data.samples))[:self.num_fake]
        self.fake_data.samples = [self.fake_data.samples[i] for i in rand_idx]
        self.fake_data.targets = None

    def __getitem__(self, index):
        choice = np.random.choice(2)
        if choice == 0: # add uniform noise
            img, _ = self.normal_data[index]
            img = torch.empty_like(img).uniform_(-self.adv_eps,
                self.adv_eps).add_(img).clamp_(0, 1)
        elif choice == 1: # fake
            rand_idx = np.random.randint(self.num_fake)
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
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = dset.CIFAR10
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        dataset = dset.CIFAR100

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=1, length=16))
        train_data = dataset(args.data_path, train=True,
                             transform=train_transform, download=True)
        train_sampler = DistributedSampler(train_data) if args.distributed else None
        train_loader = torch.utils.data.DataLoader(train_data,
            batch_size=args.batch_size//2, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        ood_train_data = OODData(args, transform_last=transforms.Normalize(mean, std))
        ood_train_sampler = DistributedSampler(ood_train_data) if args.distributed else None
        ood_train_loader = torch.utils.data.DataLoader(ood_train_data,
            batch_size=args.batch_size//4, shuffle=(ood_train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=ood_train_sampler)

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_data = dataset(args.data_path, train=False,
                            transform=test_transform, download=True)
        test_sampler = DistributedSampler(test_data) if args.distributed else None
        test_loader = torch.utils.data.DataLoader(test_data,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=test_sampler)

        adv_sampler = DistributedSampler(test_data) if args.distributed else None
        adv_loader = torch.utils.data.DataLoader(test_data,
            batch_size=args.batch_size//2, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=adv_sampler)

        fake_dataset = dset.ImageFolder(os.path.join(args.data_path_fake, 'val'),
                                        test_transform)
        fake_sampler = DistributedSampler(fake_dataset) if args.distributed else None
        fake_loader = torch.utils.data.DataLoader(fake_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=fake_sampler)

        fake_dataset2 = dset.ImageFolder(os.path.join(args.data_path_fake, 'val_extra'),
                                         test_transform)
        fake_sampler2 = DistributedSampler(fake_dataset2) if args.distributed else None
        fake_loader2 = torch.utils.data.DataLoader(fake_dataset2,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=fake_sampler2)

    else: assert False, 'Do not support dataset : {}'.format(args.dataset)

    return train_loader, ood_train_loader, test_loader, adv_loader, \
        fake_loader, fake_loader2
