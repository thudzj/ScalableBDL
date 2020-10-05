import colorcet as cc
import numpy as np
from matplotlib import pyplot as plt, colors, patheffects, offsetbox
from seaborn import distplot
from statsmodels.distributions.empirical_distribution import ECDF

from utils import (predictive_entropy, expected_calibration_error, confidence, accuracy, binned_kl_distance,
                   calibration_curve, linear_interpolation)


def training(results, path):
    fig, ax = plt.subplots(1, 2, figsize=(24, 9))
    # Plot training & validation accuracy values
    ax[0].plot(results['train_acc'])
    ax[0].plot(results['val_acc'])
    ax[0].set_title('Model accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], fontsize=14)

    # Plot training & validation loss values
    ax[1].plot(results['train_loss'])
    ax[1].plot(results['val_loss'])
    ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], fontsize=14)

    plt.tight_layout()
    plt.savefig(path + "training.pdf", format='pdf', dpi=1200)


def factors(factor_norms, path):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7), tight_layout=True)
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', labelsize=14, right=False, top=False)
        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_ylabel('Norm', fontsize=16)

    for index in range(factor_norms.shape[1]):
        axes[0].plot(factor_norms[:, index, 0])
        axes[1].plot(factor_norms[:, index, 1])
    plt.savefig(path + "_norms.pdf", format='pdf', dpi=1200)


def calibration(probabilities, labels, bins=20, swag=True, axis=None, label=None, linestyle='-', alpha=1.0,
                     color='crimson', path=""):
    ece, bin_confs, bin_accs, _ = calibration_curve(probabilities, labels, bins)
    bin_aces = bin_confs - bin_accs

    if axis is None:
        fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)
    else:
        ax = axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', labelsize=14, right=False, top=False)
    ax.set_xlabel('Confidence', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    if swag:
        ax.set_ylabel('Confidence - Accuracy', fontsize=16)

        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.plot(bin_confs, bin_aces, marker='o',
                label=f"{label} | ECE: {100 * ece:.2f}%" if label is not None else None,
                linewidth=2, linestyle=linestyle, alpha=alpha, color=color)

        ax.set_xscale('logit')
        ax.set_xlim(0.1, 0.999999)
        ax.minorticks_off()
        plt.xticks([0.2, 0.759, 0.927, 0.978, 0.993, 0.998, 0.999999],
                   labels=[0.2, 0.759, 0.927, 0.978, 0.993, 0.998, 1])

        if label is not None:
            ax.legend(fontsize=16, frameon=False)
    else:
        ax.set_ylim(0.2, 1)
        ax.plot(ax.get_xlim(), ax.get_ylim(), color='black', linestyle='dashed', linewidth=1, dashes=(5, 10))
        ax.plot(bin_confs, bin_accs, color='blueviolet', marker='o', linewidth=2)
    if axis is None:
        plt.savefig(path + "_calibration.pdf", format='pdf', dpi=1200)


def adversarial_results(epsilons, stats, bnn_stats, path):
    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)

    plt.plot(epsilons, stats["ent"], color='dodgerblue', linewidth=3, label='Deterministic')
    plt.plot(epsilons, bnn_stats["ent"], color='crimson', linewidth=3, label='Laplace')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle='dashed')
    ax.tick_params(direction='out', labelsize=14, right=False, top=False)
    ax.set_ylabel('Predictive Entropy', fontsize=16)
    ax.set_xlabel('Step size', fontsize=16)
    ax.set_xlim(0, 1)
    plt.legend(fontsize=16)
    plt.savefig(path + "_adv_entropy.pdf", format='pdf', dpi=1200)

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    plt.plot(epsilons, stats["acc"], color='dodgerblue', linewidth=3, label='Deterministic')
    plt.plot(epsilons, bnn_stats["acc"], color='crimson', linewidth=3, label='Laplace')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle='dashed')
    ax.tick_params(direction='out', labelsize=14, right=False, top=False)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_xlabel('Step size', fontsize=16)
    ax.set_xlim(0, 1)
    plt.legend(fontsize=16)
    plt.savefig(path + "_adv_accuracy.pdf", format='pdf', dpi=1200)

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    plt.plot(epsilons, stats["nll"], color='dodgerblue', linewidth=3, label='Deterministic')
    plt.plot(epsilons, bnn_stats["nll"], color='crimson', linewidth=3, label='Laplace')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle='dashed')
    ax.tick_params(direction='out', labelsize=14, right=False, top=False)
    ax.set_ylabel('NLL', fontsize=16)
    ax.set_xlabel('Step size', fontsize=16)
    ax.set_xlim(0, 1)
    plt.legend(fontsize=16)
    plt.savefig(path + "_adv_loss.pdf", format='pdf', dpi=1200)

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    plt.plot(epsilons, stats["ece1"], color='dodgerblue', linewidth=3, label='Deterministic')
    plt.plot(epsilons, bnn_stats["ece1"], color='crimson', linewidth=3, label='Laplace')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle='dashed')
    ax.tick_params(direction='out', labelsize=14, right=False, top=False)
    ax.set_ylabel('ECE', fontsize=16)
    ax.set_xlabel('Step size', fontsize=16)
    ax.set_xlim(0, 1)
    plt.legend(fontsize=16)
    plt.savefig(path + "_adv_ece.pdf", format='pdf', dpi=1200)


def inv_ecdf_vs_pred_entropy(probabilities, label=None, color='b', linestyle='-', axis=None):
    pred_ent = predictive_entropy(probabilities)
    ecdf = ECDF(pred_ent)
    x_lim = np.log(probabilities.shape[1])
    entropy_range = np.linspace(0.0, x_lim, probabilities.shape[1] * 100)
    if axis is None:
        fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)
    else:
        ax = axis
    ax.plot(entropy_range, 1 - ecdf(entropy_range), c=color, ls=linestyle, lw=3, label=label, clip_on=False)
    ax.set_xlim(ax.get_xlim()[0], np.ceil(x_lim))
    ax.set_ylim(ax.get_ylim()[0], 1)
    ax.tick_params(direction='out', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', labelsize=14, right=False, top=False)
    ax.set_ylabel('1-ecdf', fontsize=16)
    ax.set_xlabel('Predictive Entropy', fontsize=16)


def true_false_ecdf(probabilities, labels, path="", axis=None):
    true_preds = probabilities[labels == np.argmax(probabilities, axis=1)]
    false_preds = probabilities[labels != np.argmax(probabilities, axis=1)]
    true_ent = predictive_entropy(true_preds)
    false_ent = predictive_entropy(false_preds)
    true_ecdf = ECDF(true_ent)
    false_ecdf = ECDF(false_ent)

    x_lim = np.log(probabilities.shape[1])
    entropy_range = np.linspace(0.0, x_lim, probabilities.shape[1] * 100)
    if axis is None:
        fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)
    else:
        ax = axis
    ax.plot(entropy_range, 1 - true_ecdf(entropy_range), color='blueviolet', linewidth=2,
            label="Correct classification")
    ax.plot(entropy_range, 1 - false_ecdf(entropy_range), color='blueviolet', linestyle='--', linewidth=2,
            label="Misclassification")
    ax.tick_params(direction='out', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', labelsize=14, right=False, top=False)
    ax.set_ylabel('1-ecdf', fontsize=16)
    ax.set_xlabel('Predictive Entropy', fontsize=16)
    if axis is None:
        ax.legend(fontsize=16)
        plt.savefig(path if path else 'true_false_ecdf.pdf', format='pdf', dpi=1200)


def reliability_diagram(probabilities, labels, path="", bins=10, axis=None):
    ece, bin_aces, bin_accs, bin_confs = expected_calibration_error(probabilities, labels, bins=bins)
    if axis is None:
        text = offsetbox.AnchoredText(
            f"ECE: {(ece * 100):.2f}%\nAccuracy: {accuracy(probabilities, labels):.2f}%\nConfidence: {100 * confidence(probabilities):.2f}%",
            loc="upper left", frameon=False, prop=dict(fontsize=12))
        fig, ax = plt.subplots(figsize=(9, 9), tight_layout=True)
        ax.add_artist(text)
    else:
        ax = axis
    ax.bar(x=np.arange(0, 1, 0.1), height=bin_accs, width=0.1, linewidth=1, edgecolor='black', align='edge',
           color='dodgerblue')
    ax.bar(x=np.arange(0, 1, 0.1), height=bin_aces, bottom=bin_accs, width=0.1, linewidth=1, edgecolor='crimson',
           align='edge', color='crimson', fill=False, hatch='/')
    ax.bar(x=np.arange(0, 1, 0.1), height=bin_aces, bottom=bin_accs, width=0.1, linewidth=1, edgecolor='crimson',
           align='edge', color='crimson', alpha=0.3)
    if axis is None:
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.plot(ax.get_xlim(), ax.get_ylim(), color='black', linestyle='dashed', linewidth=1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', labelsize=12, right=False, top=False)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_xlabel('Confidence', fontsize=14)
        plt.savefig(path if path else 'reliability_diagram.pdf', format='pdf', dpi=1200)
    else:
        ax.tick_params(right=False, left=False, top=False, bottom=False, labelright=False, labelleft=False,
                       labeltop=False, labelbottom=False)
        ax.set_frame_on(False)


def confidence_hist(probabilities, labels=None, path=""):
    _confidence = confidence(probabilities, mean=False)
    weights = np.ones_like(_confidence) / len(_confidence)
    mean_confidence = np.mean(_confidence)

    fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True)
    ax.hist(_confidence, bins=20, edgecolor='black', linewidth=1, weights=weights, color='dodgerblue')
    conf_line = 0.72
    conf_text = 1.1
    if labels is not None:
        mean_accuracy = accuracy(probabilities, labels)
        if mean_confidence > mean_accuracy:
            acc_line = conf_line
            acc_text = 1.1
            conf_line = 0.69
            conf_text = acc_text
        else:
            acc_line = 0.8
            acc_text = 1.3
        ax.vlines(mean_accuracy, ymin=0, ymax=acc_line, linestyles='dashed')
        ax.scatter(mean_accuracy, acc_line, s=30, edgecolor='black', facecolor='white', marker='o', linewidth=1.5)
        ax.text(mean_accuracy, acc_text, f"Accuracy: {100 * mean_accuracy:.2f}%", rotation=45, verticalalignment='top',
                fontsize=14)
    ax.vlines(mean_confidence, ymin=0, ymax=conf_line, linestyles='dashed')
    ax.scatter(mean_confidence, conf_line, s=30, edgecolor='black', facecolor='white', marker='o', linewidth=1.5)
    ax.text(mean_confidence, conf_text, f"Confidence: {100 * mean_confidence:.2f}%", rotation=45,
            verticalalignment='top',
            fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.tick_params(direction='out', labelsize=14, right=False, top=False)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.set_xlabel('Confidence', fontsize=16)
    plt.savefig(path if path else 'confidence_hist.pdf', format='pdf', dpi=1200)


def entropy_hist(inclass,
                 outclass,
                 bins=100,
                 norm=True,
                 log=False,
                 kde=False,
                 jsd=False,
                 path="",
                 axis=None):
    """Makes a predictive entropy histogram or density plot for in- and out-of-domain data.

    Args:
        inclass (Numpy array): The predicted probabilities of the in-domain data.
        outclass (Numpy array): The predicted probabilities of the out-of-domain data.
        bins (int, optional): The number of bins to use for the histogram. Default: 100.
        norm (bool, optional): If True, entropy values are normalized between 0 and 1.
        log (bool, optional): If True, the x-axis is shown in log-scale. Default: False.
        kde (bool, optional): If True, plots a density instead of a histogram. Default: True.
        jsd (bool, optional): If True, calculates and prints the symmetric, discretized Kullback-Leibler divergence.
        path (string, optional): Where to save the figure. Default: Current directory.
        axis (matplotlib.Axis, optional): If provided, plots the figure on this axis.
    """
    if axis is None:
        fig, ax = plt.subplots(figsize=(9, 9), tight_layout=True)
    else:
        ax = axis

    inclass_entropy = predictive_entropy(inclass)
    outclass_entropy = predictive_entropy(outclass)
    xlim = np.log(inclass.shape[1])
    if norm:
        inclass_entropy /= xlim
        outclass_entropy /= xlim
        xlim = 1
    bins = np.linspace(0, xlim, num=bins)

    kwargs = dict(hist_kws={'alpha': .5}, kde_kws={'linewidth': 3})
    if kde:
        ax = distplot(inclass_entropy, color='dodgerblue', label='In Class', bins=bins, hist=False, ax=ax, **kwargs)
        ax = distplot(outclass_entropy, color='crimson', label='Out of Class', bins=bins, hist=False, ax=ax, **kwargs)

        l1 = ax.lines[0]
        l2 = ax.lines[1]

        x1 = l1.get_xydata()[:, 0]
        y1 = l1.get_xydata()[:, 1]
        x2 = l2.get_xydata()[:, 0]
        y2 = l2.get_xydata()[:, 1]
        ax.fill_between(x1, y1, color="dodgerblue", alpha=0.5)
        ax.fill_between(x2, y2, color="crimson", alpha=0.5)

        ax.set_ylim(0.0, ax.get_ylim()[1])
        ax.set_ylabel('Density', fontsize=16)
    else:
        kwargs['hist_kws']['histtype'] = 'stepfilled'
        kwargs['hist_kws']['edgecolor'] = 'black'
        distplot(inclass_entropy, color='dodgerblue', label='In Class', bins=bins, hist=True,
                 kde=False, ax=ax, **kwargs)
        distplot(outclass_entropy, color='crimson', label='Out of Class', bins=bins, hist=True,
                 kde=False, ax=ax, **kwargs)
        ax.set_ylabel('Frequency', fontsize=16)

    ax.set_xlim(0, xlim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', labelsize=14, right=False, top=False)
    ax.legend(fontsize=16, loc='upper right', frameon=False)
    ax.set_xlabel('Entropy', fontsize=16)

    if log:
        ax.set_xscale('log')
        if not kde:
            ax.set_xlim(np.min(bins), np.max(bins))
    if jsd:
        jsd = binned_kl_distance(inclass, outclass)
        text = offsetbox.AnchoredText(f"JSD: {jsd:.3f}", loc="upper center", frameon=True, prop=dict(fontsize=16))
        ax.add_artist(text)

    if axis is None:
        plt.savefig(path if path else "entropy_hist.pdf", format='pdf', dpi=1200)


def eigenvalue_histogram(eigenvalues: np.array,
                         remove_largest=0,
                         remove_smallest=0,
                         show_stats=False,
                         path="") -> None:
    """Plots a color-coded histogram of the provided eigenvalues.

    Args:
        eigenvalues (Numpy array): Eigenvalues to visualize.
        remove_largest (optional): Number (int) or percentage (float) of largest eigenvalues that should be removed.
        remove_smallest (optional): Number (int) or percentage (float) of smallest eigenvalues that should be removed.
        show_stats (bool, optional): Show legend with statistics like number of visualized eigenvalues.
        path (string, optional): The path where the image should be saved. Defaults to current directory.
    """
    if remove_largest > 0 or remove_smallest > 0:
        eigenvalues.sort()
        if remove_largest:
            if remove_largest >= 1:
                print(f"Removing largest {remove_largest} eigenvalues")
                eigenvalues = eigenvalues[:-remove_largest]
            else:
                print(f"Removing largest {100 * remove_largest:.0f}% of eigenvalues")
                eigenvalues = eigenvalues[:-int(len(eigenvalues) * remove_largest)]
        if remove_smallest:
            if remove_smallest >= 1:
                print(f"Removing smallest {remove_smallest} eigenvalues")
                eigenvalues = eigenvalues[remove_smallest:]
            else:
                print(f"Removing smallest {100 * remove_smallest:.0f}% of eigenvalues")
                eigenvalues = eigenvalues[int(len(eigenvalues) * remove_smallest):]

    print(f"Making histogram of {len(eigenvalues)} eigenvalues.")
    fig, ax = plt.subplots(figsize=(9, 6), tight_layout=True)
    n, bins, patches = ax.hist(eigenvalues, log=True, bins=50, edgecolor='black', linewidth=1)
    cmap = plt.cm.get_cmap('jet')
    norm = colors.Normalize(min(eigenvalues), max(eigenvalues))
    for bin, patch in zip(bins, patches):
        color = cmap(norm(bin))
        patch.set_facecolor(color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', labelsize=14, right=False, top=False)
    ax.set_xlabel('Value', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)

    if show_stats:
        num_vals = len(eigenvalues)
        num_zero = np.sum(np.array(eigenvalues) == 0.0)

        plt.text(1, 1, f"{num_vals} eigenvalues\n{num_zero} zero ({100 * (num_zero / num_vals):.0f}%)",
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax.transAxes)
    fig.savefig(path if path else "eigenvalue_hist.pdf", format='pdf', dpi=1200)


def hyper_results(data, metric='acc', annotate=True, top=3, path=""):
    norms, scales = data['norm'][data['score'] < 200], data['scale'][data['score'] < 200]
    color_data = data[metric][data['score'] < 200] if metric == 'score' else data[metric]

    if metric == 'acc':
        best = np.argsort(color_data)[-top:]
        cmap = cc.cm.bmy
    else:
        best = np.argsort(color_data)[:top]
        cmap = cc.cm.rainbow

    sizes = linear_interpolation(30, 250, data=np.arange(len(color_data)))
    fig, ax = plt.subplots(figsize=(9, 9), tight_layout=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', labelsize=14, right=False, top=False)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', linewidth=.5, zorder=0)
    ax.set_axisbelow(True)
    sc = ax.scatter(norms, scales, edgecolors='black', c=color_data, cmap=cmap, alpha=.8, s=sizes, linewidth=1)
    ax.scatter(data['norm'][data['score'] == 200], data['scale'][data['score'] == 200], c='r', marker='x', s=20)
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_axisbelow(True)
    cbar.ax.set_title(metric.upper(), fontsize=14)
    ax.set_ylabel(r'$\log N$', fontsize=16)
    ax.set_xlabel(r'$\log\tau$', fontsize=16)
    if annotate:
        ax.scatter(norms[best], scales[best], facecolors='none', edgecolors='r', marker='s', s=200)
        for x, y, text in zip(norms[best], scales[best], color_data[best]):
            signA = np.random.choice([-1, 1])
            angleA = 0
            xtext = np.random.randint(40, 70)
            signB = np.random.choice([-1, 1])
            angleB = np.random.choice([30, 45, 90])
            txt = ax.annotate(f"{text:.3f} [{x:.3f}, {y:.3f}]", xy=(x, y), xycoords='data',
                              xytext=(signA * xtext, signB * (xtext - 20)),
                              textcoords='offset points', fontsize=10, color='white', zorder=100,
                              arrowprops=dict(arrowstyle="-|>",
                                              connectionstyle=f"angle3,angleA={signA * angleA},angleB={signB * angleB}"))
            txt.set_path_effects([patheffects.Stroke(linewidth=1, foreground='black'), patheffects.Normal()])
    plt.savefig(path if path else "hyper_results.pdf", format='pdf', dpi=1200)


def hyper_results_3d(data, metric='acc', path=""):
    fig, ax = plt.subplots(figsize=(9, 9), tight_layout=True)
    norms, scales = data['norm'][data['score'] < 200], data['scale'][data['score'] < 200]
    color_data = data[metric][data['score'] < 200] if metric == 'score' else data[metric]

    CS = plt.tricontourf(norms, scales, color_data, cmap=cc.cm.bmy, levels=np.arange(0, 100, 5))
    # plt.clabel(CS, inline=1, fontsize=10)
    plt.show()
