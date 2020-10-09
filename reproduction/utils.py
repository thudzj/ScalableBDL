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
    return -(p*p.log()).sum(-1)

def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
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
        raise NotImplementedError

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

    ap = metrics.average_precision_score(y, x)

    # Plot formatting
    plt.legend()#(prop={'size': 20})
    plt.xlabel('Mutual information')#, fontsize=20)
    plt.ylabel('Density')#, fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_, '{}_vs_{}.pdf'.format('nat'
        if type2_ is None else type2_, type_)), bbox_inches='tight')
    return ap

def plot_ens(dir_, rets, baseline_acc):
    lw = 1.25
    color = ['red', 'green', 'darkorange', 'b']
    if isinstance(rets, list):
        rets = np.stack([np.array(item) for item in rets])
    min_acc = min(rets[:, 2].min(), rets[:, 6].min(), baseline_acc) - 0.1
    max_acc = max(rets[:, 2].max(), rets[:, 6].max(), baseline_acc) + 0.1

    fig = plt.figure(figsize=(4,3))
    fig, ax1 = plt.subplots(figsize=(4,3))
    l1 = ax1.plot(rets[:, 0]+1, rets[:, 2], color=color[0], lw=lw, alpha=0.6)
    l2 = ax1.plot(rets[:, 0]+1, rets[:, 6], color=color[1], lw=lw)
    l3 = ax1.plot(rets[:, 0]+1, np.ones(rets.shape[0])*baseline_acc,
        color=color[2], lw=lw, alpha=0.6, linestyle='dashed')
    ax1.set_yticks(np.arange(1, 101, 1))
    ax1.set_xticks([1,] + list(np.arange(20, rets.shape[0]+1, 20)))
    ax1.set_ylim((min_acc, max_acc))
    ax1.set_xlim((1, rets.shape[0]))
    ax1.set_xlabel('The number of MC sample')
    ax1.set_ylabel('Test accuracy (%)')
    ax1.legend(l1+l2+l3, ['Individual', 'Ensemble', 'Deterministic'],
        loc = 'best', fancybox=True, columnspacing=0.5, handletextpad=0.2,
        borderpad=0.15) # +l3+l4 , 'Indiv ECE', 'Ensemble ECE'  , fontsize=11
    plt.savefig(os.path.join(dir_, 'ens_plot.pdf'), format='pdf',
        dpi=600, bbox_inches='tight')


#------------------------for face verification------------------------
# MIT License
# Copyright (c) 2016 David Sandberg

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn
from scipy import interpolate
from scipy.spatial.distance import pdist

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds = 10, pca = 0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits = nrof_folds, shuffle = False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
        # dist = pdist(np.vstack([embeddings1, embeddings2]), 'cosine')

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print("doing pca on", fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis = 0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components = pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
#         print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds = 10):
    '''
    Copy from [insightface](https://github.com/deepinsight/insightface)
    :param thresholds:
    :param embeddings1:
    :param embeddings2:
    :param actual_issame:
    :param far_target:
    :param nrof_folds:
    :return:
    '''
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits = nrof_folds, shuffle = False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind = 'slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def verify(embeddings, actual_issame, nrof_folds = 10, pca = 0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings1, embeddings2, np.asarray(actual_issame), nrof_folds = nrof_folds, pca = pca)
#     thresholds = np.arange(0, 4, 0.001)
#     val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
#                                       np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
#     return tpr, fpr, accuracy, best_thresholds, val, val_std, far
    return tpr, fpr, accuracy, best_thresholds
