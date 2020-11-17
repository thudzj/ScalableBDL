import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import seaborn as sns

from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})

weights = torch.load("/data/zhijie/snapshots_ba/emp-nores-blur.03-6epoch-mineps.5-2/checkpoint.pth.tar", map_location='cpu')['state_dict']

bayes_weights = []
for k, v in weights.items():
    if 'weights' in k or 'biases' in k:
        print(k, v.shape)
        bayes_weights.append(v.view(20, -1).data.cpu().numpy())

bayes_weights = np.concatenate(bayes_weights, 1)

bayes_weights = PCA(n_components=20).fit_transform(bayes_weights)

gram = cosine_similarity(bayes_weights, bayes_weights)
print(bayes_weights.shape, gram)


with sns.axes_style("white"):
    ax = sns.heatmap(gram, mask=None, vmax=1., square=True,  cmap="YlGnBu")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('posterior_sim.pdf', bbox_inches='tight')
