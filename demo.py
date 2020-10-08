import argparse
from tqdm import tqdm
import torch
from converter import to_bayesian, to_deterministic
from utils import freeze, unfreeze
from optimizers import PsiSGD

import sys
sys.path.insert(0, './reproduction')
from dataset.cifar import load_dataset
from models.wrn import wrn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 1
    args.dataset = 'cifar10'
    args.data_path = './data'
    args.cutout = True
    args.distributed = False
    args.batch_size = 4
    args.workers = 4
    train_loader, test_loader = load_dataset(args)

    net = wrn(pretrained=True, depth=28, width=10).cuda()
    for m in net.modules():
        if m.__class__.__name__.startswith('Dropout'): 
            m.eval()
    bayesian_net = to_bayesian(net)

    mus, psis = [], []
    for name, param in bayesian_net.named_parameters():
        if 'psi' in name: 
            psis.append(param)
        else: 
            mus.append(param)
    mu_optimizer = torch.optim.SGD([{'params': mus, 'weight_decay': 2e-4}], 
                lr=0.0008, momentum=0.9, nesterov=True)
    psi_optimizer = PsiSGD([{'params': psis, 'weight_decay': 2e-4}],
                lr=0.1, momentum=0.9, nesterov=True, num_data=50000)

    bayesian_net.train()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    for epoch in range(args.epochs):
        for i, (input, target) in enumerate(train_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = bayesian_net(input)
            loss = criterion(output, target)

            mu_optimizer.zero_grad()
            psi_optimizer.zero_grad()
            loss.backward()
            if i % 100 == 0:
                print("Epoch {}, ite {}/{}, loss {}".format(epoch, i, 
                    len(train_loader), loss.item()))
            mu_optimizer.step()
            psi_optimizer.step()
    
    