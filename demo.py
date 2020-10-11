import argparse
from tqdm import tqdm
import torch
from torch.optim import SGD
from scalablebdl.bnn_utils import freeze, unfreeze, disable_dropout, Bayes_ensemble
from scalablebdl.mean_field import PsiSGD, to_bayesian, to_deterministic

import sys
sys.path.insert(0, './exps')
from dataset.cifar import load_dataset
from models.wrn import wrn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 1
    args.dataset = 'cifar10'
    args.data_path = '/data/LargeData/Regular/cifar'
    args.cutout = True
    args.distributed = False
    args.batch_size = 32
    args.workers = 4
    train_loader, test_loader = load_dataset(args)

    net = wrn(pretrained=True, depth=28, width=10).cuda()
    disable_dropout(net)

    eval_loss, eval_acc = Bayes_ensemble(test_loader, net,
                                         num_mc_samples=1)
    print('Results of deterministic pre-training, '
          'eval loss {}, eval acc {}'.format(eval_loss, eval_acc))

    net.stage_3[-1].conv2 = to_bayesian(net.stage_3[-1].conv2)
    net.lastact = to_bayesian(net.lastact)
    net.classifier = to_bayesian(net.classifier)
    unfreeze(net)

    mus, psis = [], []
    for name, param in net.named_parameters():
        if 'psi' in name: psis.append(param)
        else: mus.append(param)
    mu_optimizer = SGD(mus, lr=0.0008, momentum=0.9,
                       weight_decay=2e-4, nesterov=True)
    psi_optimizer = PsiSGD(psis, lr=0.1, momentum=0.9,
                           weight_decay=2e-4, nesterov=True,
                           num_data=50000)

    for epoch in range(args.epochs):
        net.train()
        for i, (input, target) in enumerate(train_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = net(input)
            loss = torch.nn.functional.cross_entropy(output, target)

            mu_optimizer.zero_grad()
            psi_optimizer.zero_grad()
            loss.backward()
            mu_optimizer.step()
            psi_optimizer.step()

            if i % 100 == 0:
                print("Epoch {}, ite {}/{}, loss {}".format(epoch, i,
                    len(train_loader), loss.item()))

        eval_loss, eval_acc = Bayes_ensemble(test_loader, net)
        print("Epoch {}, eval loss {}, eval acc {}".format(
            epoch, eval_loss, eval_acc))
