# A plug-and-play implementation for *Bayesian fine-tuning* to practically learn Bayesian Neural Networks
---
We provide a Pytorch implementation to learn Bayesian Neural Networks (BNNs) at low cost. We unfold the learning of a BNN into two steps: *deterministic pre-training* of the deep neural network (DNN) counterpart of the BNN followed by *Bayesian fine-tuning*.


For *deterministic pre-training*, we just train a regular DNN via *maximum a posteriori* (MAP) estimation, which is realised by performing SGD under weight decay regularizor. We can also reuse off-the-shelf pre-trained models from popular model zoos (e.g., [PyTorch Hub](https://pytorch.org/hub/)).


After *deterministic pre-training*, it is straight forward to convert the converged DNN into a BNN, and then perform *Bayesian fine-tuning* given this library, which enables us to fine-tune the BNN as if one were fine-tuning a regular DNN.

The current implementation only considers using mean-field Gaussian as approximate posterior, and more flexible distributions are under development.

For more details, refer to [our paper](https://arxiv.org/pdf/2010.01979.pdf) and [GitHub page](https://thudzj.github.io/ScalableBDL/).



## Usage
### Dependencies
+ python 3
+ torch 1.3.0
+ torchvision 0.4.1

### Installation
+ `pip install git+https://github.com/thudzj/ScalableBDL.git`


### A fast trial
With CIFAR-10 classification as an example, we leverage this library to perform *Bayesian fine-tuning* upon a pre-trained wide-ResNet-28-10 model, and to evaluate the resultant Bayesian posterior following the following procedure:

We first import the necessary modules for *Bayesian fine-tuning*:
```python
from scalablebdl.converter import to_bayesian, to_deterministic
from scalablebdl.bnn_utils import freeze, unfreeze, disable_dropout, Bayes_ensemble
from scalablebdl.mean_field import PsiSGD
```

Then load the pre-trained wide-ResNet-28-10 model, and disable the possible stochastic components inside the model:
```python
net = wrn(pretrained=True, depth=28, width=10).cuda()
disable_dropout(net)
```

We can check the performence of such a model by one-sample Bayes ensemble as it is deterministic:
```python
eval_loss, eval_acc = Bayes_ensemble(test_loader, net,
                                     num_mc_samples=1)
print('Results of deterministic pre-training, '
      'eval loss {}, eval acc {}'.format(eval_loss, eval_acc))
```

To expand the point-estimate parameters into random variables, we only need to invoke
```python
bayesian_net = to_bayesian(net)
bayesian_net.apply(unfreeze)
```

To realise fine-tuning, we build two optimizers with inherent weight decay modules for the mean and variance of the approximate posterior:
```python
mus, psis = [], []
for name, param in bayesian_net.named_parameters():
    if 'psi' in name: psis.append(param)
    else: mus.append(param)
mu_optimizer = torch.optim.SGD(mus, lr=0.0008, momentum=0.9, 
                               weight_decay=2e-4, nesterov=True)
psi_optimizer = PsiSGD(psis, lr=0.1, momentum=0.9, 
                       weight_decay=2e-4, nesterov=True, 
                       num_data=50000)
```

Note that the optimizer for `psi` takes one more argument `num_data`, which equals to the size of the training dataset.


After these preparation, we perform *Bayesian fine-tuning* just like training a regular DNN, expect that our optimization involves two optimizers:
```python
for epoch in range(args.epochs):
    bayesian_net.train()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = bayesian_net(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        mu_optimizer.zero_grad()
        psi_optimizer.zero_grad()
        loss.backward()
        mu_optimizer.step()
        psi_optimizer.step()

        if i % 100 == 0:
            print("Epoch {}, ite {}/{}, loss {}".format(epoch, i,
                len(train_loader), loss.item()))

    eval_loss, eval_acc = Bayes_ensemble(test_loader, bayesian_net)
    print("Epoch {}, eval loss {}, eval acc {}".format(
        epoch, eval_loss, eval_acc))
```
 
** Check [this](https://github.com/thudzj/ScalableBDL/blob/master/demo.py) for a complete and runnable script.**


## Thanks to
* @Harry24k [github:bayesian-neural-network-pytorch](https://github.com/Harry24k/bayesian-neural-network-pytorch)

## Contact and coorperate
If you have any problem about this library or want to contribute to it, please send us an Email at:
- dzj17@mails.tsinghua.edu.cn

## Cite
Please cite our paper if you use this code in your own work:
```
@inproceedings{
deng2021bayesadapter,
title={BayesAdapter: Being Bayesian, Inexpensively and Robustly, via Bayeisan Fine-tuning},
author={Deng, Zhijie and Xiao, Yang and Hao, Zhang and Yinpeng, Dong and Zhu, Jun},
booktitle={ArXiv},
year={2021},
url={https://arxiv.org/pdf/2010.01979.pdf},
}
```

