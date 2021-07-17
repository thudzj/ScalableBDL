import argparse
from tqdm import tqdm
import torch
from torch.optim import SGD
from scalablebdl.bnn_utils import freeze, unfreeze, disable_dropout, Bayes_ensemble
from scalablebdl.prior_reg import PriorRegularizor
from scalablebdl.mean_field import PsiSGD, to_bayesian, to_deterministic

import sys
sys.path.insert(0, './reproduction')
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
    args.num_mc_samples = 20
    train_loader, test_loader = load_dataset(args)

    net = wrn(pretrained=True, depth=28, width=10).cuda()
    disable_dropout(net)

    eval_loss, eval_acc = Bayes_ensemble(test_loader, net,
                                         num_mc_samples=1)
    print('Results of deterministic pre-training, '
          'eval loss {}, eval acc {}'.format(eval_loss, eval_acc))

    bayesian_net = to_bayesian(net)
    unfreeze(bayesian_net)

    mus, psis = [], []
    for name, param in bayesian_net.named_parameters():
        if 'psi' in name: psis.append(param)
        else: mus.append(param)
    optimizer = SGD([{"params": mus, "lr": 0.0008, "weight_decay": 2e-4},
                     {"params": psis, "lr": 0.1, "weight_decay": 0}],
                    momentum=0.9, nesterov=True)
    regularizer = PriorRegularizor(bayesian_net, decay=2e-4, num_data=50000, 
                                   num_mc_samples=args.num_mc_samples)

    for epoch in range(args.epochs):
        bayesian_net.train()
        for i, (input, target) in enumerate(train_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = bayesian_net(input)
            loss = torch.nn.functional.cross_entropy(output, target)

            optimizer.zero_grad()
            loss.backward()
            regularizer.step()
            optimizer.step()

            if i % 100 == 0:
                print("Epoch {}, ite {}/{}, loss {}".format(epoch, i,
                    len(train_loader), loss.item()))

        eval_loss, eval_acc = Bayes_ensemble(test_loader, bayesian_net)
        print("Epoch {}, eval loss {}, eval acc {}".format(
            epoch, eval_loss, eval_acc))
