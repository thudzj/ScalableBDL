import argparse
from tqdm import tqdm
import torch
from converter import to_bayesian, to_deterministic
from utils import freeze, unfreeze
from optimizers import psi_opt

import sys
sys.path.insert(0, './reproduction')
from dataset.cifar import load_dataset
from models.wrn import wrn

def acc(model, loader):
    model.eval()
    num_true = 0
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(loader), total=100):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            num_true += (model(input).argmax(1) == target).int().sum().item()
    return num_true/float(len(test_loader.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = 'cifar10'
    args.data_path = './data'
    args.cutout = True
    args.distributed = False
    args.batch_size = 64
    args.workers = 4
    train_loader, test_loader = load_dataset(args)

    net = wrn(pretrained=True, depth=28, width=10).cuda()
    # print(net)
    print(acc(net, test_loader))

    bayesian_net = to_bayesian(net)
    # print(bayesian_net)
    print(acc(bayesian_net, test_loader))

    bayesian_net.apply(freeze)
    print(acc(bayesian_net, test_loader))

    deterministic_net = to_deterministic(bayesian_net)
    # print(deterministic_net)
    print(acc(deterministic_net, test_loader))
    
    