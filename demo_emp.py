import argparse
from tqdm import tqdm
import torch
from torch.optim import SGD
from scalablebdl.bnn_utils import set_mc_sample_id, disable_dropout
from scalablebdl.empirical import to_bayesian, to_deterministic

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
    args.batch_size = 64
    args.workers = 4
    args.num_mc_samples = 20
    train_loader, test_loader = load_dataset(args)

    net = wrn(pretrained=True, depth=28, width=10).cuda()
    disable_dropout(net)

    net.stage_3[-1].conv2 = to_bayesian(net.stage_3[-1].conv2, args.num_mc_samples)
    net.lastact = to_bayesian(net.lastact, args.num_mc_samples)
    net.classifier = to_bayesian(net.classifier, args.num_mc_samples)

    pretrained_params, one_layer_bayes_params = [], []
    for name, param in net.named_parameters():
        if 'weights' in name or 'biases' in name:
            one_layer_bayes_params.append(param)
        else:
            pretrained_params.append(param)

    pretrained_optimizer = SGD(pretrained_params, lr=0.0008, momentum=0.9,
                               weight_decay=2e-4, nesterov=True)
    one_layer_bayes_optimizer = SGD(one_layer_bayes_params, lr=0.1, momentum=0.9,
                                    weight_decay=2e-4/args.num_mc_samples, nesterov=True)

    for epoch in range(args.epochs):
        net.train()
        for i, (input, target) in enumerate(train_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            set_mc_sample_id(net, args.num_mc_samples, batch_size=input.size(0))
            output = net(input)
            loss = torch.nn.functional.cross_entropy(output, target)

            pretrained_optimizer.zero_grad()
            one_layer_bayes_optimizer.zero_grad()
            loss.backward()
            pretrained_optimizer.step()
            one_layer_bayes_optimizer.step()

            if i % 100 == 0:
                print("Epoch {}, ite {}/{}, loss {}".format(epoch, i,
                    len(train_loader), loss.item()))

        eval_loss, eval_acc = Bayes_ensemble(test_loader, net)
        print("Epoch {}, eval loss {}, eval acc {}".format(
            epoch, eval_loss, eval_acc))
