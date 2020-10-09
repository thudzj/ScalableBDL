from .mean_field import BayesLinearMF, BayesConv2dMF, BayesBatchNorm2dMF

def freeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)):
        m.deterministic = True

def unfreeze(m):
    if isinstance(m, (BayesConv2dMF, BayesLinearMF, BayesBatchNorm2dMF)):
        m.deterministic = False

def disable_dropout(net):
    for m in net.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p = 0
