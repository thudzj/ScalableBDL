import os
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from .utils import GeneratedDataAugment

def load_dataset(args):
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

    train_sampler = DistributedSampler(train_dataset) if args.distributed else None

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

    val_sampler = DistributedSampler(val_dataset) if args.distributed else None

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader


class OODData:
    def __init__(self, args, transform_last=None):
        super(OODData, self).__init__()
        self.eps = args.epsilon
        self.transform_last = transform_last
        self.num_fake = args.num_fake

        self.normal_data = dset.ImageFolder(os.path.join(args.data_path, 'train'),
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]))
        self.fake_data = dset.ImageFolder(os.path.join(args.data_path_fake, 'train'),
            transforms.Compose([
                GeneratedDataAugment(args),
                transforms.RandomCrop(224),
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
            img = torch.empty_like(img).uniform_(-self.eps,
                self.eps).add_(img).clamp_(0, 1)
        elif choice == 1: # fake
            rand_idx = np.random.randint(self.num_fake)
            img, _ = self.fake_data[rand_idx]
        else:
            raise NotImplementedError
        img = self.transform_last(img)
        return img

    def __len__(self):
        return len(self.normal_data)

def load_dataset_ft(args):
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
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size//2, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    ood_train_dataset = OODData(args, transform_last=normalize)
    ood_train_sampler = DistributedSampler(ood_train_dataset) if args.distributed else None
    ood_train_loader = torch.utils.data.DataLoader(
        ood_train_dataset, batch_size=args.batch_size//4, shuffle=(ood_train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=ood_train_sampler)

    val_dataset = dset.ImageFolder(
        os.path.join(args.data_path, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_sampler = DistributedSampler(val_dataset) if args.distributed else None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    adv_sampler = DistributedSampler(val_dataset) if args.distributed else None
    adv_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size//2, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=adv_sampler)

    fake_dataset = dset.ImageFolder(
        os.path.join(args.data_path_fake, 'val'),
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    fake_sampler = DistributedSampler(fake_dataset) if args.distributed else None
    fake_loader = torch.utils.data.DataLoader(
        fake_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=fake_sampler)

    adv_dataset2 = dset.ImageFolder(
        args.data_path_adv, #"adv_samples/fgsm_resnet_152"
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    adv_sampler2 = DistributedSampler(adv_dataset2) if args.distributed else None
    adv_loader2 = torch.utils.data.DataLoader(
        adv_dataset2, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=adv_sampler2)

    return train_loader, ood_train_loader, val_loader, adv_loader, fake_loader, adv_loader2
