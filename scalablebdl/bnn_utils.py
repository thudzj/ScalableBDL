import numpy as np
import torch
from .mean_field import BayesLinearMF, BayesConv2dMF, BayesBatchNorm2dMF
from .empirical import BayesLinearEMP, BayesConv2dEMP, BayesBatchNorm2dEMP
from .implicit import BayesLinearIMP, BayesConv2dIMP, BayesBatchNorm2dIMP

# freeze and unfreeze work for mean-field and implicit posteriors
def freeze(net):
    net.apply(_freeze)

def _freeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)) \
            or isinstance(m, (BayesConv2dIMP, BayesLinearIMP, BayesBatchNorm2dIMP)):
        m.deterministic = True

def unfreeze(net):
    net.apply(_unfreeze)

def _unfreeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)) \
            or isinstance(m, (BayesConv2dIMP, BayesLinearIMP, BayesBatchNorm2dIMP)):
        m.deterministic = False

# set_mode only works for empirical posterior
def set_mode(net, mode=None, batch_size=None, num_modes=20):
    if mode is None:
        mode = np.random.randint(0, num_modes, size=batch_size)
    else:
        if isinstance(mode, int):
            assert mode >= 0 and mode < num_modes, "Mode must be in [0, num_modes)"
        else:
            assert isinstance(mode, np.array)
    for m in net.modules():
        if isinstance(m, (BayesConv2dEMP, BayesLinearEMP, BayesBatchNorm2dEMP)):
            m.mode = mode

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
                set_mode(model, j)
                output += model(input).softmax(-1)
            output /= num_mc_samples
            total_loss += loss_metric(output.log(), target).item()
            total_acc += acc_metric(output, target).item()
        total_loss /= len(loader)
        total_acc /= len(loader)
    return total_loss, total_acc
