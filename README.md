# A plug-and-play implementation for *Bayesian fine-tuning* to practically learn Bayesian Neural Networks
---
We provide a Pytorch implementation to learn Bayesian Neural Networks (BNNs) at low cost. We unfold the learning of a BNN into two steps: *deterministic pre-training* of the deep neural network (DNN) counterpart of the BNN followed by *Bayesian fine-tuning*.


For *deterministic pre-training*, we just train a regular DNN via *maximum a posteriori* (MAP) estimation, which is realised by conducting optimization under weight decay regularizor. We can also reuse off-the-shelf pre-trained models from popular model zoos (e.g., [PyTorch Hub](https://pytorch.org/hub/)).


After *deterministic pre-training*, it is straight forward to convert the converged DNN into a BNN and to perform *Bayesian fine-tuning* given this library.

The current implementation only considers using mean-field Gaussian as approximate posterior, and more flexible distributions are under development.

For more details, refer to our [BayesAdapter paper](https://arxiv.org/pdf/2010.01979.pdf) and [GitHub page](https://thudzj.github.io/ScalableBDL/).

## Bayesian fine-tuning based approach for adversarial detection
We apply the Bayesian fine-tuning paradigm to detect adversarial examples, and observe promising results. See our [LiBRe paper](https://arxiv.org/pdf/2103.14835.pdf) for more details. To reproduce LiBRe, check [here](https://github.com/thudzj/ScalableBDL/tree/efficient/exps).

## Usage
### Dependencies
+ python 3
+ torch 1.3.0+
+ torchvision 0.4.1+

### Installation
+ `pip install git+https://github.com/thudzj/ScalableBDL.git`


### A fast trial
With CIFAR-10 classification as an example, we can easily leverage this library to perform *Bayesian fine-tuning* upon a pre-trained wide-ResNet-28-10 model, and to evaluate the resultant Bayesian posterior.

We first import the necessary modules for *Bayesian fine-tuning*:
```python
from scalablebdl.bnn_utils import freeze, unfreeze, disable_dropout, Bayes_ensemble
from scalablebdl.prior_reg import PriorRegularizor
from scalablebdl.mean_field import PsiSGD, to_bayesian, to_deterministic
```

Then load the pre-trained wide-ResNet model, and disable the possible stochasticity inside the model:
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

To expand the point-estimate parameters into Bayesian variables, we only need to invoke
```python
bayesian_net = to_bayesian(net, num_mc_samples=args.num_mc_samples)
unfreeze(bayesian_net)
```

To realise fine-tuning, we build two optimizers with inherent weight decay modules for the mean and variance of the approximate posterior:
```python
mus, psis = [], []
for name, param in bayesian_net.named_parameters():
    if 'psi' in name: psis.append(param)
    else: mus.append(param)
optimizer = SGD([{"params": mus, "lr": 0.0008, "weight_decay": 2e-4},
                 {"params": psis, "lr": 0.1, "weight_decay": 0}],
                momentum=0.9, nesterov=True)
regularizer = PriorRegularizor(bayesian_net, decay=2e-4, num_data=50000,
                               num_mc_samples=args.num_mc_samples)
```

The `regularizer` absorbs the KL divergence between the approximate posterior and the prior.


After the preparation, we perform *Bayesian fine-tuning* just like fine-tuning a regular DNN, expect that our optimization involves two optimizers:
```python
for epoch in range(args.epochs):
    bayesian_net.train()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = bayesian_net(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
	regularizer.step()
        optimizer.step()

        if i % 100 == 0:
            print("Epoch {}, ite {}/{}, loss {}".format(epoch, i,
                len(train_loader), loss.item()))

    eval_loss, eval_acc = Bayes_ensemble(test_loader, bayesian_net)
    print("Epoch {}, eval loss {}, eval acc {}".format(
        epoch, eval_loss, eval_acc))
```
 
**Check [this](https://github.com/thudzj/ScalableBDL/blob/master/demo.py) for a complete and runnable script.**


## Comparison on predictive performance
We compare the predictive performance between the fine-tuning start point (*DNN*) and the obtained *BNN* in the following table. Note that we perform *Bayes ensemble* with 100 MC samples for estimating the accuracy of *BNN*.

||CIFAR-10 (wide-ResNet-28-10)|ImageNet (ResNet-50)|
| :-------------: | :----------: | :-----------: |
|*DNN*|96.92%|76.13%|
|*BNN*|**97.09%**|**76.49%**|

## Thanks to
* @Harry24k [github:bayesian-neural-network-pytorch](https://github.com/Harry24k/bayesian-neural-network-pytorch)

## Contact and cooperate
If you have any problem about this library or want to contribute to it, please send us an Email at:
- dzj17@mails.tsinghua.edu.cn

## Cite
Please cite our paper if you use this code in your own work:
```
@article{deng2020bayesadapter,
  title={BayesAdapter: Being Bayesian, Inexpensively and Reliably, via Bayesian Fine-tuning},
  author={Deng, Zhijie and Zhang, Hao and Yang, Xiao and Dong, Yinpeng and Zhu, Jun},
  journal={arXiv preprint arXiv:2010.01979},
  year={2020}
}

@inproceedings{deng2021libre,
  title={LiBRe: A Practical Bayesian Approach to Adversarial Detection},
  author={Deng, Zhijie and Yang, Xiao and Xu, Shizhen and Su, Hang and Zhu, Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={972--982},
  year={2021}
}
```

