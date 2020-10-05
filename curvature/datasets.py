import os
import random
import multiprocessing as mp
import ctypes

import numpy as np
import torch
import torchvision
import tqdm
from PIL import Image
from matplotlib import pyplot as plt

from utils import setup, ram, vram
Image.MAX_IMAGE_PIXELS = 688290000
torch.manual_seed(0)


def test_loader():
    args = setup()

    print("Loading data")
    if args.data == 'cifar10':
        data = cifar10(args.torch_data, splits='test')
    if args.data == 'mnist':
        data = mnist(args.torch_data, splits='test')
    elif args.data == 'tiny':
        data = imagenet(args.data_dir, img_size=64, batch_size=args.batch_size, splits='test', tiny=True)
    elif args.data == 'imagenet':
        img_size = 224
        if args.model in ['googlenet', 'inception_v3']:
            img_size = 299
        data = imagenet(args.data_dir, img_size, args.batch_size, workers=args.workers, splits='train')
    elif args.data == 'gtsrb':
        data = gtsrb(args.data_dir, batch_size=args.batch_size, workers=args.workers, splits='train')

    for epoch in range(args.epochs):
        data = tqdm.tqdm(data, desc=f"Epoch [{epoch + 1}/{args.epochs}]")
        for batch, (images, labels) in enumerate(data):
            data.set_postfix({'RAM': ram(), 'VRAM': vram()})
            fig, ax = plt.subplots(5, 12)
            i = 0
            j = 0
            for img in images:
                img = np.moveaxis(img.numpy(), 0, -1)
                img = (img - np.min(img)) / np.ptp(img)
                ax[i, j].imshow(img)
                ax[i, j].axis('off')
                j += 1
                if j > 0 and j % 12 == 0:
                    j = 0
                    i += 1
            plt.show()
            plt.close()


def fgsm(model, images, labels, criterion=torch.nn.CrossEntropyLoss(), epsilon=0.1):
    vmin, vmax = images.min(), images.max()
    images.requires_grad = True

    logits = model(images)
    loss = criterion(logits, labels)
    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_image = images + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, vmin, vmax)

    return perturbed_image


class BGR(object):
    def __call__(self, pic):
        img = np.array(pic)[..., ::-1]
        return Image.fromarray(img)


class Binarize(object):
    def __call__(self, pic):
        return Image.fromarray(np.uint8(np.random.binomial(1, np.array(pic) / 255) * 255))


class PCA(object):
    def __init__(self, scale=0.1):
        self.scale = scale
        self.eigval = torch.tensor([0.2175, 0.0188, 0.0045])
        self.eigvec = torch.tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])

    def __call__(self, img):
        alpha = img.new().resize_(3).normal_(0, self.scale)
        rgb = self.eigvec.type_as(img).clone().mul(alpha.view(1, 3).expand(3, 3)).mul(
            self.eigval.view(1, 3).expand(3, 3)).sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


class RandomInterpolation(object):
    def __init__(self, size):
        self.size = size
        self.methods = [
            torchvision.transforms.RandomResizedCrop(size=self.size, interpolation=Image.NEAREST),
            torchvision.transforms.RandomResizedCrop(size=self.size, interpolation=Image.BILINEAR),
            torchvision.transforms.RandomResizedCrop(size=self.size, interpolation=Image.BICUBIC),
            torchvision.transforms.RandomResizedCrop(size=self.size, interpolation=Image.ANTIALIAS),
            torchvision.transforms.RandomResizedCrop(size=self.size, interpolation=Image.LINEAR),
            torchvision.transforms.RandomResizedCrop(size=self.size, interpolation=Image.LANCZOS),
            torchvision.transforms.RandomResizedCrop(size=self.size, interpolation=Image.CUBIC)
        ]

    def __call__(self, pic):
        return random.choice(self.methods)(pic)


class Memory(torch.utils.data.Dataset):
    def __init__(self, data, img_size=224, channels=3):
        self.data = data
        self.images = torch.zeros(len(data), channels, img_size, img_size)
        self.targets = torch.zeros(len(data)).long()
        self.use_cache = False

    def pin_memory(self):
        self.images = self.images.pin_memory()
        self.targets = self.targets.pin_memory()
        return self

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def __getitem__(self, index):
        if self.use_cache:
            if index < 1:
                return self.images, self.targets
            else:
                raise StopIteration()
        else:
            self.images[index] = self.data[index][0]
            self.targets[index] = self.data[index][1]
            return self.images[index], self.targets[index]

    def __len__(self):
        if self.use_cache:
            return 1
        else:
            return len(self.data)


class Cashed(torch.utils.data.Dataset):
    def __init__(self, data, img_size=224, channels=3):
        self.data = data
        shared_array_base = mp.Array(ctypes.c_float, len(data) * channels * img_size ** 2)
        shared_array_base_labels = mp.Array(ctypes.c_long, len(data))
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array_labels = np.ctypeslib.as_array(shared_array_base_labels.get_obj())
        shared_array = shared_array.reshape(len(data), channels, img_size, img_size)
        self.shared_array = torch.from_numpy(shared_array)
        self.shared_array_labels = torch.from_numpy(shared_array_labels).long()
        self.use_cache = False

    def pin_memory(self):
        self.shared_array = self.shared_array.pin_memory()
        self.shared_array_labels = self.shared_array_labels.pin_memory()
        return self

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def __getitem__(self, index):
        if not self.use_cache:
            self.shared_array[index] = self.data[index][0]
            self.shared_array_labels[index] = self.data[index][1]
        return self.shared_array[index], self.shared_array_labels[index]

    def __len__(self):
        return len(self.data)


def mnist(root, batch_size=32, workers=6, augment=True, splits=('train', 'val')):
    val_transform = torchvision.transforms.ToTensor()
    if augment:
        transform = torchvision.transforms.Compose([Binarize(), torchvision.transforms.ToTensor()])
    else:
        transform = val_transform

    loader_list = list()
    if 'train' in splits:
        train_set = torchvision.datasets.MNIST(root, train=True, transform=transform, download=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                   num_workers=workers, pin_memory=True)
        loader_list.append(train_loader)
    if 'test' in splits or 'val' in splits:
        val_test_set = torchvision.datasets.MNIST(root, train=False, transform=val_transform, download=True)
        val_set, test_set = torch.utils.data.random_split(val_test_set, [5000, 5000])

        if 'val' in splits:
            val_set = Memory(val_set, img_size=28, channels=1)
            for _ in val_set:
                pass
            val_set.set_use_cache(True)
            val_set.pin_memory()
            loader_list.append(val_set)

        if 'test' in splits:
            test_set = Memory(test_set, img_size=28, channels=1)
            for _ in test_set:
                pass
            test_set.set_use_cache(True)
            test_set.pin_memory()
            loader_list.append(test_set)

    if len(loader_list) == 1:
        return loader_list[0]
    return loader_list


def kmnist(root, batch_size=32, workers=6, splits=('train', 'val')):
    loader_list = list()
    if 'train' in splits or 'val' in splits:
        train_val_set = torchvision.datasets.KMNIST(root, train=True, download=True,
                                                    transform=torchvision.transforms.ToTensor())

        val_set, train_set = torch.utils.data.random_split(train_val_set, [10000, len(train_val_set) - 10000])
        if 'train' in splits:
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                       num_workers=workers, pin_memory=True)
            loader_list.append(train_loader)
        if 'val' in splits:
            val_set = Memory(val_set, img_size=28, channels=1)
            for _ in val_set:
                pass
            val_set.set_use_cache(True)
            val_set.pin_memory()
            loader_list.append(val_set)
    if 'test' in splits:
        test_set = torchvision.datasets.KMNIST(root, train=False, download=True,
                                               transform=torchvision.transforms.ToTensor())
        test_set = Memory(test_set, img_size=28, channels=1)
        for _ in test_set:
            pass
        test_set.set_use_cache(True)
        test_set.pin_memory()
        loader_list.append(test_set)

    if len(loader_list) == 1:
        return loader_list[0]
    return loader_list


def cifar10(root, batch_size=32, workers=6, augment=True, splits=('train', 'val')):
    normalize = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    val_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])

    if augment:
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.RandomCrop(32, padding=4),
             torchvision.transforms.RandomHorizontalFlip(),
             # torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
             # torchvision.transforms.RandomRotation(degrees=5),
             torchvision.transforms.ToTensor(),
             normalize])
    else:
        transform = val_transform

    loader_list = list()
    if 'train' in splits:
        train_val_set = torchvision.datasets.CIFAR10(root, train=True, transform=transform, download=True)
        train_loader = torch.utils.data.DataLoader(train_val_set, batch_size=batch_size, shuffle=True,
                                                   num_workers=workers, pin_memory=True)
        loader_list.append(train_loader)
    if 'test' in splits or 'val' in splits:
        val_test_set = torchvision.datasets.CIFAR10(root, train=False, transform=val_transform, download=True)
        val_set, test_set = torch.utils.data.random_split(val_test_set, [5000, 5000])

        if 'val' in splits:
            val_set = Memory(val_set, img_size=32, channels=3)
            for _ in val_set:
                pass
            val_set.set_use_cache(True)
            val_set.pin_memory()
            loader_list.append(val_set)

        if 'test' in splits:
            test_set = Memory(test_set, img_size=32, channels=3)
            for _ in test_set:
                pass
            test_set.set_use_cache(True)
            test_set.pin_memory()
            loader_list.append(test_set)

    if len(loader_list) == 1:
        return loader_list[0]
    return loader_list


def svhn(root, batch_size=32, workers=6, splits=('train', 'val')):
    normalize = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])

    loader_list = list()
    if 'train' in splits:
        train_set = torchvision.datasets.SVHN(root, split='train', transform=transform, download=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                       num_workers=workers, pin_memory=True)
        loader_list.append(train_loader)
    if 'test' in splits or 'val' in splits:
        val_test_set = torchvision.datasets.SVHN(root, split='test', transform=transform, download=True)
        val_set, test_set, rest = torch.utils.data.random_split(val_test_set, [5000, 5000, len(val_test_set) - 10000])

        if 'val' in splits:
            val_set = Memory(val_set, img_size=32, channels=3)
            for _ in val_set:
                pass
            val_set.set_use_cache(True)
            val_set.pin_memory()
            loader_list.append(val_set)

        if 'test' in splits:
            test_set = Memory(test_set, img_size=32, channels=3)
            for _ in test_set:
                pass
            test_set.set_use_cache(True)
            test_set.pin_memory()
            loader_list.append(test_set)

    if len(loader_list) == 1:
        return loader_list[0]
    return loader_list


def art(root, img_size=224, batch_size=32, workers=6, pin_memory=True, use_cache=False, pre_cache=False):
    test_dir = os.path.join(root, "not_imagenet/data")

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(img_size * 8 / 7)),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_set = torchvision.datasets.ImageFolder(test_dir, transform)
    if use_cache:
        test_set = Cashed(test_set, img_size, channels=3)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=workers,
                                              pin_memory=pin_memory)
    if use_cache and pre_cache:
        print("Caching")
        for _ in tqdm.tqdm(test_loader):
            pass
        test_loader.dataset.set_use_cache(True)
        # test_loader.dataset.pin_memory()
    return test_loader


def imagenet(root, img_size=224, batch_size=32, workers=6, splits=('train', 'val'), tiny=False, pin_memory=True,
             use_cache=False, pre_cache=False):
    if tiny:
        path = os.path.join(root, "imagenet/data/tiny-imagenet-200")
    else:
        path = os.path.join(root, "imagenet/data")
    train_dir = os.path.join(path, 'train')
    test_dir = os.path.join(path, 'val')

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transform_list = list()
    if not tiny:
        val_transform_list.append(torchvision.transforms.Resize(int(img_size * 8 / 7)))
        val_transform_list.append(torchvision.transforms.CenterCrop(img_size))
    val_transform_list.append(torchvision.transforms.ToTensor())
    val_transform_list.append(normalize)
    val_transform = torchvision.transforms.Compose(val_transform_list)

    train_transform_list = list()
    if tiny:
        train_transform_list.append(torchvision.transforms.RandomCrop(img_size, padding=8))
    else:
        train_transform_list.append(torchvision.transforms.RandomResizedCrop(img_size))
    train_transform_list.append(torchvision.transforms.RandomHorizontalFlip())
    train_transform_list.append(torchvision.transforms.ToTensor())
    train_transform_list.append(normalize)
    train_transform = torchvision.transforms.Compose(train_transform_list)

    loader_list = list()
    if 'train' in splits:
        train_val_set = torchvision.datasets.ImageFolder(train_dir, train_transform)
        train_loader = torch.utils.data.DataLoader(train_val_set, batch_size=batch_size, shuffle=True,
                                                   num_workers=workers, pin_memory=pin_memory)
        loader_list.append(train_loader)

    if 'val' or 'test' in splits:
        val_test_set = torchvision.datasets.ImageFolder(test_dir, val_transform)
        val_set, test_set = torch.utils.data.random_split(val_test_set, [25000, 25000])

        if 'test' in splits:
            if use_cache:
                test_set = Cashed(test_set, img_size, channels=3)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=workers,
                                                      pin_memory=pin_memory)
            if use_cache and pre_cache:
                print("Caching")
                for _ in tqdm.tqdm(test_loader):
                    pass
                test_loader.dataset.set_use_cache(True)
                # test_loader.dataset.pin_memory()
            loader_list.append(test_loader)

        if 'val' in splits:
            if use_cache:
                val_set = Cashed(val_set, img_size, channels=3)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=workers,
                                                     pin_memory=pin_memory)
            if use_cache and pre_cache:
                print("Caching")
                for _ in tqdm.tqdm(val_loader):
                    pass
                val_loader.dataset.set_use_cache(True)
                # val_loader.dataset.pin_memory()
            loader_list.append(val_loader)

    if len(loader_list) == 1:
        return loader_list[0]


def gtsrb(root, img_size=32, batch_size=32, workers=6, splits=('train', 'val'), pin_memory=True):
    train_dir = os.path.join(root, "GTSRB/Training/Images")
    val_dir = os.path.join(root, "GTSRB/Validation/Images")
    test_dir = os.path.join(root, "GTSRB/Test/Images")

    normalize = torchvision.transforms.Normalize([0.34038433, 0.3119956, 0.32119358],
                                                 [0.05087305, 0.05426421, 0.05859348])
    if img_size > 32:
        val_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(int(img_size * 8 / 7)),
                                                        torchvision.transforms.CenterCrop(img_size),
                                                        torchvision.transforms.ToTensor(),
                                                        normalize])
        train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(img_size),
                                                          torchvision.transforms.RandomAffine(degrees=15,
                                                                                              translate=(0.1, 0.1),
                                                                                              shear=10),
                                                          torchvision.transforms.ColorJitter(brightness=0.1,
                                                                                             contrast=0.1,
                                                                                             saturation=0.1,
                                                                                             hue=0.1),
                                                          torchvision.transforms.ToTensor(),
                                                          normalize])
    else:
        val_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size + 10),
                                                        torchvision.transforms.CenterCrop(img_size),
                                                        torchvision.transforms.ToTensor(),
                                                        normalize])
        train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(img_size, padding=4),
                                                          torchvision.transforms.RandomAffine(degrees=15,
                                                                                              translate=(0.1, 0.1),
                                                                                              scale=(0.9, 1.1),
                                                                                              shear=10),
                                                          torchvision.transforms.ColorJitter(brightness=0.1,
                                                                                             contrast=0.1,
                                                                                             saturation=0.1,
                                                                                             hue=0.1),
                                                          torchvision.transforms.ToTensor(),
                                                          normalize])

    loader_list = list()
    if 'train' in splits:
        train_set = torchvision.datasets.ImageFolder(train_dir, train_transform)

        weights = list()
        for c in range(43):
            dir_name = f"000{c}" if c > 9 else f"0000{c}"
            weights.append(len(os.listdir(os.path.join(train_dir, dir_name))[:-1]))
        weights = 1 / np.array(weights)
        weights = np.array([weights[t] for t in train_set.targets])
        sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.from_numpy(weights).double(), len(weights))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                                                   num_workers=workers, pin_memory=pin_memory)
        loader_list.append(train_loader)
    if 'val' in splits:
        val_set = torchvision.datasets.ImageFolder(val_dir, val_transform)
        if img_size > 32:
            val_set = Cashed(val_set, img_size, channels=3)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=workers,
                                                     pin_memory=pin_memory)
            for _ in val_loader:
                pass
            val_loader.dataset.set_use_cache(True)
            val_loader.dataset.pin_memory()
            loader_list.append(val_loader)
        else:
            val_set = Memory(val_set, img_size=img_size, channels=3)
            for _ in val_set:
                pass
            val_set.set_use_cache(True)
            val_set.pin_memory()
            loader_list.append(val_set)

    if 'test' in splits:
        test_set = torchvision.datasets.ImageFolder(test_dir, val_transform)
        test_set = Memory(test_set, img_size=img_size, channels=3)
        for _ in test_set:
            pass
        test_set.set_use_cache(True)
        test_set.pin_memory()
        loader_list.append(test_set)

    if len(loader_list) == 1:
        return loader_list[0]
    return loader_list


if __name__ == "__main__":
    test_loader()
