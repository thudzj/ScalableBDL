import sys, os
import numpy as np
from sklearn import metrics

lines = open(sys.argv[1]).readlines()

stats = {}

flag = False
for line in lines:
    if '--> auroc' in line:
        if flag == False:
            flag = True

        if flag:
            attack = line.split(" ")[2]
            auroc = float(line.split(" ")[5].replace(";", ""))

            if attack in stats:
                stats[attack].append(auroc)
            else:
                stats[attack] = [auroc,]
    elif flag == True:
        break

if len(stats.items()) == 0:
    dir_ = '/'.join(sys.argv[1].split('/')[:-1])
    for attack in ['FGSM', 'BIM', 'PGD', 'MIM', 'TIM', 'DI_MIM', 'FGSM_L2', 'BIM_L2', 'PGD_L2']:
        for dset in ['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30', 'calfw', 'cplfw', 'vgg2_fp']:
            mi_nat = np.load(os.path.join(dir_, 'mis_{}.npy'.format(dset)))
            mi_adv = np.load(os.path.join(dir_, 'mis_{}_{}.npy'.format(attack, dset)))
            x = np.concatenate((mi_nat, mi_adv), 0)
            y = np.zeros(x.shape[0])
            y[mi_nat.shape[0]:] = 1

            auroc = metrics.roc_auc_score(y, x)
            fpr, tpr, thresholds = metrics.roc_curve(y, x)
            accs = {th: tpr[np.argwhere(fpr <= th).max()] for th in [0.01, 0.05, 0.1]}

            if attack in stats:
                stats[attack].append(auroc)
            else:
                stats[attack] = [auroc,]

for k, v in stats.items():
    print(k, np.mean(v), len(v))
