import numpy as np
import torch
from .mean_field import BayesLinearMF, BayesConv2dMF, BayesBatchNorm2dMF

# freeze and unfreeze work for mean-field and implicit posteriors
def freeze(net):
    net.apply(_freeze)

def _freeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)):
        m.deterministic = True

def unfreeze(net):
    net.apply(_unfreeze)

def _unfreeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)):
        m.deterministic = False

def disable_dropout(net):
    for m in net.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p = 0

def Bayes_ensemble(loader, model, loss_metric=torch.nn.functional.cross_entropy,
                   acc_metric=lambda arg1, arg2: (arg1.argmax(-1)==arg2).float().mean(),
                   num_mc_samples=20):
    model.eval()
    with torch.no_grad():
        total_loss, total_acc = 0, 0
        for i, (input, target) in enumerate(loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = 0
            for j in range(num_mc_samples):
                output += model(input).softmax(-1)
            output /= num_mc_samples
            total_loss += loss_metric(output.log(), target).item()
            total_acc += acc_metric(output, target).item()
        total_loss /= len(loader)
        total_acc /= len(loader)
    return total_loss, total_acc
