import sys, os
import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 22})


import seaborn as sns


dsets = ['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30', 'calfw', 'cplfw', 'vgg2_fp']


fig = plt.figure()
ax = plt.subplot(111)

dir_ = sys.argv[1] #'/data/zhijie/snapshots_ba/emp-nores-blur.03-6epoch-mineps.5-2'
attacks = ['FGSM', 'BIM', 'PGD', 'MIM', 'TIM', 'DI_MIM', 'FGSM_L2', 'BIM_L2', 'PGD_L2', 'CW']
mis = {}

if 'face' in dir_:
    mis['Normal'] = np.concatenate([np.load(os.path.join(dir_, 'mis_{}.npy'.format(dset))) for dset in dsets])
else:
    mis['Normal'] = np.load(os.path.join(dir_, 'mis.npy'))

max_ = 0
for attack in attacks:
    if 'face' in dir_:
        if attack == 'CW':
            continue
        mis[attack] = np.concatenate([np.load(os.path.join(dir_, 'mis_{}_{}.npy'.format(attack, dset))) for dset in dsets])
    else:
        mis[attack] = np.load(os.path.join(dir_, 'mis_{}.npy'.format(attack)))


    mis[attack] = mis[attack][~np.isnan(mis[attack])]
    max_ = max(max_, mis[attack].max())

    x = np.concatenate((mis['Normal'], mis[attack]), 0)
    y = np.zeros(x.shape[0])
    y[mis['Normal'].shape[0]:] = 1
    auroc = metrics.roc_auc_score(y, x)
    print(attack, auroc)

# Draw the density plot
for k, v in mis.items():
    sns.distplot(v, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 1, 'clip': (-0.0001, 1.)},
                 label = k if k != 'DI_MIM' else 'DIM')

# Plot formatting
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=11)
#
#
# ax = plt.subplot(142)
# dir_ = '/data/zhijie/snapshots_ba/mfg-nores-nou-6epoch'
# attacks = ['FGSM', 'CW', 'BIM', 'PGD', 'MIM', 'TIM', 'DI_MIM', 'FGSM_L2', 'BIM_L2', 'PGD_L2']
# mis = {}
# mis['Normal'] = np.load(os.path.join(dir_, 'mis.npy'))
# max_ = 0
# for attack in attacks:
#     mis[attack] = np.load(os.path.join(dir_, 'mis_{}.npy'.format(attack)))
#     mis[attack] = mis[attack][~np.isnan(mis[attack])]
#     max_ = max(max_, mis[attack].max())
# # Draw the density plot
# for k, v in mis.items():
#     sns.distplot(v, hist = False, kde = True,
#                  kde_kws = {'shade': True, 'linewidth': 1, 'clip': (-0.0001, max_*1.1)},
#                  label = k)

ax.get_legend().remove()
# plt.legend()#(prop={'size': 20})
plt.xlabel('Feature variance uncertainty')#, fontsize=20)
plt.ylabel('Density')#, fontsize=20)

xlim_ = 1.
if dir_ == '/data/zhijie/snapshots_ba/emp-nores-blur.03-6epoch-mineps.5-2':
    xlim_ = 0.5
elif dir_ == '/data/zhijie/snapshots_ba/emp-nores-nou-6epoch':
    xlim_ = 0.1
elif dir_ == '/data/zhijie/snapshots_ba/face_ir50_softmax-alpha100':
    xlim_ = 0.3
elif dir_ == '/data/zhijie/snapshots_ba/face_ir50_softmax-mcd':
    xlim_ = 1.

if xlim_ < 0.5:
    plt.xticks(np.arange(1, 6)/5.* xlim_, ['{:.2f}'.format(gg) for gg in np.arange(1, 6)/5.*xlim_])
else:
    plt.xticks(np.arange(1, 6)/5.* xlim_, ['{:.1f}'.format(gg) for gg in np.arange(1, 6)/5.*xlim_])
plt.xlim([0, xlim_])
plt.tight_layout()
# plt.savefig(os.path.join(dir_, 'density_all.pdf'), bbox_inches='tight')
plt.savefig(dir_.split('/')[-1] + 'density_all.pdf', bbox_inches='tight')
