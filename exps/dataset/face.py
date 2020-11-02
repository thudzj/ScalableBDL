import os
import bcolz
import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler
from .utils import GeneratedDataAugment

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

def load_dataset(args, INPUT_SIZE=[112, 112],
                      RGB_MEAN = [0.5, 0.5, 0.5], RGB_STD = [0.5, 0.5, 0.5],
                      val_datasets=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30', 'calfw', 'cplfw', 'vgg2_fp']):
    train_transform = transforms.Compose([
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN, std = RGB_STD)
    ])
    train_data = dset.ImageFolder(os.path.join(args.data_path, 'CASIA-maxpy-align'), train_transform)
    weights = torch.DoubleTensor(make_weights_for_balanced_classes(train_data.imgs, len(train_data.classes)))
    if args.distributed:
        from catalyst.data.sampler import DistributedSamplerWrapper
        train_sampler = DistributedSamplerWrapper(
            WeightedRandomSampler(weights, len(weights)))
    else:
        train_sampler = WeightedRandomSampler(weights, len(weights))
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

def load_dataset_ft(args, INPUT_SIZE=[112, 112],
                      RGB_MEAN = [0.5, 0.5, 0.5], RGB_STD = [0.5, 0.5, 0.5],
                      val_datasets=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30', 'calfw', 'cplfw', 'vgg2_fp']):
    train_transform = transforms.Compose([
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN, std = RGB_STD)
    ])
    train_data = dset.ImageFolder(os.path.join(args.data_path,
        'CASIA-maxpy-align{}'.format('-all' if args.use_all_face_data else '')), train_transform)
    weights = torch.DoubleTensor(make_weights_for_balanced_classes(train_data.imgs, len(train_data.classes)))
    if args.distributed:
        from catalyst.data.sampler import DistributedSamplerWrapper
        train_sampler = DistributedSamplerWrapper(
            WeightedRandomSampler(weights, len(weights)))
    else:
        train_sampler = WeightedRandomSampler(weights, len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size = args.batch_size//2, shuffle=(train_sampler is None),
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
