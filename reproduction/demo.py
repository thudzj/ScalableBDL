import argparse
from tqdm import tqdm
import torch
from torch.optim import SGD
from scalablebdl.converter import to_bayesian, to_deterministic
from scalablebdl.bnn_utils import freeze, unfreeze, disable_dropout
from scalablebdl.mean_field import PsiSGD

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
    disable_dropout(net)
    bayesian_net = to_bayesian(net)
    bayesian_net.apply(unfreeze)

    mus, psis = [], []
    for name, param in bayesian_net.named_parameters():
        if 'psi' in name: psis.append(param)
        else: mus.append(param)
    mu_optimizer = SGD(mus, 0.0008, 0.9, weight_decay=2e-4, nesterov=True)
    psi_optimizer = PsiSGD(psis, 0.1, 0.9, weight_decay=2e-4,
                           nesterov=True, num_data=50000)

    bayesian_net.train()
    for epoch in range(args.epochs):
        for i, (input, target) in enumerate(train_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = bayesian_net(input)
            loss = torch.nn.functional.cross_entropy(output, target)

            mu_optimizer.zero_grad()
            psi_optimizer.zero_grad()
            loss.backward()
            mu_optimizer.step()
            psi_optimizer.step()

            if i % 100 == 0:
                print("Epoch {}, ite {}/{}, loss {}".format(epoch, i,
                    len(train_loader), loss.item()))
