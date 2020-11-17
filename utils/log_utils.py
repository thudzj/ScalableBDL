import torch
import torch.distributed as dist
import os, sys, time
import numpy as np
import pandas as pd

import skimage.transform
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})

import seaborn as sns
from sklearn import metrics

def ent(p):
    return -(p*((p+1e-10).log())).sum(-1)

def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t().contiguous()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0)

def print_log(print_string, log, force=False):
    if log[1] == 0 or force:
        print("{}".format(print_string))
        log[0].write('{}\n'.format(print_string))
        log[0].flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

class _ECELoss(torch.nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

        bin_boundaries_plot = torch.linspace(0, 1, 11)
        self.bin_lowers_plot = bin_boundaries_plot[:-1]
        self.bin_uppers_plot = bin_boundaries_plot[1:]

    def forward(self, confidences, predictions, labels, title):
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        accuracy_in_bin_list = []
        for bin_lower, bin_upper in zip(self.bin_lowers_plot, self.bin_uppers_plot):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            accuracy_in_bin = 0
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean().item()
            accuracy_in_bin_list.append(accuracy_in_bin)

        fig = plt.figure(figsize=(4,3))
        p1 = plt.bar(np.arange(10) / 10., accuracy_in_bin_list, 0.1, align = 'edge', edgecolor ='black')
        p2 = plt.plot([0,1], [0,1], '--', color='gray')

        plt.ylabel('Accuracy', fontsize=14)
        plt.xlabel('Confidence', fontsize=14)
        #plt.title(title)
        plt.xticks(np.arange(0, 1.01, 0.2))
        plt.yticks(np.arange(0, 1.01, 0.2))
        plt.xlim(left=0,right=1)
        plt.ylim(bottom=0,top=1)
        plt.grid(True)
        #plt.legend((p1[0], p2[0]), ('Men', 'Women'))
        plt.text(0.1, 0.83, 'ECE: {:.4f}'.format(ece.item()), fontsize=14)
        plt.savefig(title, format='pdf', dpi=600, bbox_inches='tight')
        return ece

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs

def time_file_str():
  ISOTIMEFORMAT='%Y-%m-%d'
  string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string + '-{}'.format(random.randint(1, 10000))

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def refresh(self, epochs):
    if epochs == self.total_epoch: return
    self.epoch_losses = np.vstack( (self.epoch_losses, np.zeros((epochs - self.total_epoch, 2), dtype=np.float32) - 1) )
    self.epoch_accuracy = np.vstack( (self.epoch_accuracy, np.zeros((epochs - self.total_epoch, 2), dtype=np.float32)) )
    self.total_epoch = epochs

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == val_acc

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()

  def plot_curve(self, save_path):
    title = 'the accuracy/loss curve of train/val'
    dpi = 80
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)

    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)


    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
      # print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)

def plot_mi(dir_, type_, type2_=None):
    if type2_ is None:
        mi_nat = np.load(os.path.join(dir_, 'mis.npy'))
        label2_ = 'Normal'
    else:
        mi_nat = np.load(os.path.join(dir_, 'mis_{}.npy'.format(type2_)))
        label2_ = type2_#.capitalize()
    mi_svhn = np.load(os.path.join(dir_, 'mis_{}.npy'.format(type_)))
    fig = plt.figure()

    if type_ == 'advg':
        label_ = 'Adversarial'
    elif type_ == 'adv':
        label_ = 'adversarial-T'
    elif 'adv_' in type_:
        label_ = 'Adversarial'
    elif type_ == "svhn":
        label_ = 'SVHN'
    elif type_ == "celeba":
        label_ = 'CelebA'
    elif type_ == "noise":
        label_ = 'noise'
    elif type_ == "fake":
        label_ = 'Fake'
    elif type_ == "fake2":
        label_ = 'Fake'
    else:
        label_ = type_.capitalize()

    # Draw the density plot
    sns.distplot(mi_nat, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'clip': (-0.0001, 3)},
                 label = label2_)
    sns.distplot(mi_svhn, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'clip': (-0.0001, 3)},
                 label = label_)

    x = np.concatenate((mi_nat, mi_svhn), 0)
    y = np.zeros(x.shape[0])
    y[mi_nat.shape[0]:] = 1

    ap = metrics.roc_auc_score(y, x)
    fpr, tpr, thresholds = metrics.roc_curve(y, x)
    accs = {th: tpr[np.argwhere(fpr <= th).max()] for th in [0.01, 0.05, 0.1]}

    # Plot formatting
    plt.legend()#(prop={'size': 20})
    plt.xlabel('Mutual information uncertainty')#, fontsize=20)
    plt.ylabel('Density')#, fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_, '{}_vs_{}.pdf'.format('nat'
        if type2_ is None else type2_, type_)), bbox_inches='tight')
    return "auroc: {:.4f}; ".format(ap) + "; ".join(["TPR: {:.4f} @ FPR={:.4f}".format(v, k) for k, v in accs.items()])

def plot_ens(dir_, rets, baseline_acc=None):
    lw = 1.25
    color = ['red', 'green', 'darkorange', 'b']
    if isinstance(rets, list):
        rets = np.stack([np.array(item) for item in rets])

    if baseline_acc is not None:
        min_acc = min(rets[:, 2].min(), rets[:, 6].min(), baseline_acc) - 0.1
        max_acc = max(rets[:, 2].max(), rets[:, 6].max(), baseline_acc) + 0.1
    else:
        min_acc = min(rets[:, 2].min(), rets[:, 6].min()) - 0.1
        max_acc = max(rets[:, 2].max(), rets[:, 6].max()) + 0.1

    fig = plt.figure(figsize=(4,3))
    fig, ax1 = plt.subplots(figsize=(4,3))
    l1 = ax1.plot(rets[:, 0]+1, rets[:, 2], color=color[0], lw=lw, alpha=0.6)
    l2 = ax1.plot(rets[:, 0]+1, rets[:, 6], color=color[1], lw=lw)
    if baseline_acc is not None:
        l3 = ax1.plot(rets[:, 0]+1, np.ones(rets.shape[0])*baseline_acc,
            color=color[2], lw=lw, alpha=0.6, linestyle='dashed')
    ax1.set_yticks(np.arange(1, 101, 1))
    ax1.set_xticks([1,] + list(np.arange(20, rets.shape[0]+1, 20)))
    ax1.set_ylim((min_acc, max_acc))
    ax1.set_xlim((1, rets.shape[0]))
    ax1.set_xlabel('The number of MC sample')
    ax1.set_ylabel('Test accuracy (%)')
    if baseline_acc is not None:
        ax1.legend(l1+l2+l3, ['Individual', 'Ensemble', 'Deterministic'],
            loc = 'best', fancybox=True, columnspacing=0.5, handletextpad=0.2,
            borderpad=0.15) # +l3+l4 , 'Indiv ECE', 'Ensemble ECE'  , fontsize=11
    else:
        ax1.legend(l1+l2, ['Individual', 'Ensemble'],
            loc = 'best', fancybox=True, columnspacing=0.5, handletextpad=0.2,
            borderpad=0.15) # +l3+l4 , 'Indiv ECE', 'Ensemble ECE'  , fontsize=11
    plt.savefig(os.path.join(dir_, 'ens_plot.pdf'), format='pdf',
        dpi=600, bbox_inches='tight')

class NoneOptimizer():
    def __init__(self):
        self.param_groups = [{'lr': 0}]

    def zero_grad(self):
        pass

    def step(self):
        pass

    def load_state_dict(self, _):
        pass

    def state_dict(self, **kargs):
        return None

def gaussian_kernel():
    def gkern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        import scipy.stats as st

        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    stack_kernel = stack_kernel.transpose((2, 3, 0, 1))
    stack_kernel = torch.from_numpy(stack_kernel)
    return stack_kernel

def smooth(x, stack_kernel):
    ''' implemenet depthwiseConv with padding_mode='SAME' in pytorch '''
    padding = (stack_kernel.size(-1) - 1) // 2
    groups = x.size(1)
    return torch.nn.functional.conv2d(x, weight=stack_kernel, padding=padding, groups=groups)
