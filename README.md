# A plug-and-play implementation for *Bayesian fine-tuning* to practically learn Bayesian Neural Networks
---
We provide a Pytorch implementation to learn Bayesian Neural Networks (BNNs) at low cost. We unfold the learning of a BNN into two steps: *deterministic pre-training* of the deep neural network (DNN) counterpart of the BNN followed by *Bayesian fine-tuning*.


For *deterministic pre-training*, we just train a regular DNN via *maximum a posteriori* (MAP) estimation, which is realised by performing SGD under weight decay regularizor. We can also reuse off-the-shelf pre-trained models from popular model zoos (e.g., [PyTorch Hub](https://pytorch.org/hub/)).


After *deterministic pre-training*, it is straight forward to convert the converged DNN into a BNN, and then perform *Bayesian fine-tuning* given the provided implementation, which enables us to fine-tune the BNN as if one were fine-tuning a regular DNN.

For more details, refer to [our paper](https://arxiv.org/pdf/2010.01979.pdf) and [GitHub page](https://thudzj.github.io/ScalableBDL/).



## Usage
### Dependencies
+ python 3
+ torch 1.3.0
+ torchvision 0.4.1

### Installation
+ `pip install git+https://github.com/thudzj/ScalableBDL.git`


### A fast trial
As an example, we leverage this library to perform *Bayesian fine-tuning* upon the wide-ResNet-28-10 model pre-trained on CIFAR-10, and to evaluate the resultant Bayesian posterior following the following procedure:

```python
from scalablebdl.converter import to_bayesian, to_deterministic
from scalablebdl.bnn_utils import freeze, unfreeze, disable_dropout
from scalablebdl.mean_field import PsiSGD
```
 ([code](https://github.com/thudzj/ScalableBDL/blob/master/demo.py))


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

