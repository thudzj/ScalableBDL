import numpy as np
import torch
from .mean_field import BayesLinearMF, BayesConv2dMF, BayesBatchNorm2dMF, BayesPReLUMF
from .empirical import BayesLinearEMP, BayesConv2dEMP, BayesBatchNorm2dEMP, BayesPReLUEMP
from .low_rank import BayesLinearLR, BayesConv2dLR
# from .implicit import BayesLinearIMP, BayesConv2dIMP, BayesBatchNorm2dIMP, BayesPReLUIMP

# freeze and unfreeze work for mean-field and implicit posteriors
def freeze(net):
    net.apply(_freeze)

def _freeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF, BayesPReLUMF)) \
            or isinstance(m, (BayesConv2dLR, BayesLinearLR)):
        m.deterministic = True

def unfreeze(net):
    net.apply(_unfreeze)

def _unfreeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF, BayesPReLUMF)) \
            or isinstance(m, (BayesConv2dLR, BayesLinearLR)):
        m.deterministic = False

# set_mc_sample_id only works for empirical posterior
def set_mc_sample_id(net, num_mc_samples, mc_sample_id=None, batch_size=None):
    if mc_sample_id is None:
        mc_sample_id = np.random.randint(0, num_mc_samples, size=batch_size)
    else:
        if isinstance(mc_sample_id, int):
            assert mc_sample_id >= 0 and mc_sample_id < num_mc_samples, \
                "Mc_sample_id must be in [0, num_mc_samples)"
        else:
            assert isinstance(mc_sample_id, np.ndarray)
    for m in net.modules():
        if isinstance(m, (BayesConv2dEMP, BayesLinearEMP, BayesBatchNorm2dEMP, BayesPReLUEMP,
                BayesConv2dLR, BayesLinearLR)):
            m.mc_sample_id = mc_sample_id

def disable_dropout(net):
    for m in net.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p = 0

def parallel_eval(net):
    net.apply(_parallel_eval)

def _parallel_eval(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF, BayesPReLUMF)) \
            or isinstance(m, (BayesConv2dLR, BayesLinearLR)) \
            or isinstance(m, (BayesLinearEMP, BayesConv2dEMP, BayesBatchNorm2dEMP, BayesPReLUEMP)):
        m.parallel_eval = True

def disable_parallel_eval(net):
    net.apply(_disable_parallel_eval)

def _disable_parallel_eval(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF, BayesPReLUMF)) \
            or isinstance(m, (BayesConv2dLR, BayesLinearLR)) \
            or isinstance(m, (BayesLinearEMP, BayesConv2dEMP, BayesBatchNorm2dEMP, BayesPReLUEMP)):
        m.parallel_eval = False

def Bayes_ensemble(loader, model, loss_metric=torch.nn.functional.cross_entropy,
                   acc_metric=lambda arg1, arg2: (arg1.argmax(-1)==arg2).float().mean()):
    model.eval()
    parallel_eval(model)
    with torch.no_grad():
        total_loss, total_acc = 0, 0
        for i, (input, target) in enumerate(loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            outputs = model(input).softmax(-1)
            output = outputs.mean(-2)
            total_loss += loss_metric(output.log(), target).item()
            total_acc += acc_metric(output, target).item()
        total_loss /= len(loader)
        total_acc /= len(loader)
    disable_parallel_eval(model)
    return total_loss, total_acc
