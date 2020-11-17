import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils_lid import compute_roc
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics

attacks = ['normal', 'FGSM', 'CW', 'BIM', 'PGD', 'MIM', 'TIM', 'FGSM_L2', 'BIM_L2', 'PGD_L2'] #'DI_MIM'
methods = ['kd', 'lid']
features = {}


for method in methods:
    for attack in attacks:
        features["{}-{}".format(method, attack)] = np.load("/data/zhijie/snapshots_ba/lids/{}_{}.npy".format(method, attack))
        if method == 'kd':
            features["{}-{}".format(method, attack)] = features["{}-{}".format(method, attack)].reshape(-1, 1)

for method in methods:
    for train_attack in ['FGSM', 'BIM', 'PGD']:
        normal_features = features["{}-{}".format(method, 'normal')]
        attack_features = features["{}-{}".format(method, train_attack)]

        # eval on source attack
        rand_pert = np.random.permutation(normal_features.shape[0])
        normal_features = normal_features[rand_pert]
        eval_normal_features = normal_features[:normal_features.shape[0]//2]
        train_normal_features = normal_features[normal_features.shape[0]//2:]

        rand_pert = np.random.permutation(attack_features.shape[0])
        attack_features = attack_features[rand_pert]
        eval_attack_features = attack_features[:attack_features.shape[0]//2]
        train_attack_features = attack_features[attack_features.shape[0]//2:]

        if method == 'kd':
            X = np.concatenate([normal_features, attack_features])[:, 0]
            Y = np.ones(X.shape[0])
            Y[normal_features.shape[0]:] = 0
            print("{} dirct on {}, eval on {}, auroc {}".format(method, train_attack, train_attack, metrics.roc_auc_score(Y, X)))

        X_train = np.concatenate([train_normal_features, train_attack_features])
        Y_train = np.zeros(X_train.shape[0]).reshape(-1, 1)
        Y_train[train_normal_features.shape[0]:] = 1

        X_eval = np.concatenate([eval_normal_features, eval_attack_features])
        Y_eval = np.zeros(X_eval.shape[0]).reshape(-1, 1)
        Y_eval[eval_normal_features.shape[0]:] = 1

        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_eval = scaler.transform(X_eval)

        lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
        _, _, auc_score = compute_roc(Y_eval.ravel(), lr.predict_proba(X_eval)[:, 1], plot=False)

        print("{} train on {}, eval on {}, auroc {}".format(method, train_attack, train_attack, auc_score))

        for val_attack in attacks:
            if val_attack == 'normal' or val_attack == train_attack:
                continue

            # eval on transfer attack
            eval_features = features["{}-{}".format(method, val_attack)]
            rand_pert = np.random.permutation(eval_features.shape[0])
            eval_features = eval_features[rand_pert]
            eval_features_ = eval_features[:eval_features.shape[0]//2]

            if method == 'kd':
                X = np.concatenate([normal_features, eval_features])[:, 0]
                Y = np.ones(X.shape[0])
                Y[normal_features.shape[0]:] = 0
                print("{} dirct on {}, eval on {}, auroc {}".format(method, train_attack, val_attack, metrics.roc_auc_score(Y, X)))

            X_eval = np.concatenate([eval_normal_features, eval_features_])
            Y_eval = np.zeros(X_eval.shape[0]).reshape(-1, 1)
            Y_eval[eval_normal_features.shape[0]:] = 1

            X_eval = scaler.transform(X_eval)
            _, _, auc_score = compute_roc(Y_eval.ravel(), lr.predict_proba(X_eval)[:, 1], plot=False)

            print("{} train on {}, eval on {}, auroc {}".format(method, train_attack, val_attack, auc_score))
