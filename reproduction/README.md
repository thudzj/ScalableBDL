## Reproduce the results in our paper

At first, execute `pip install git+https://github.com/thudzj/ScalableBDL.git`

### Dependencies
+ bcolz
+ catalyst
+ imageio
+ matplotlib
+ opencv-python
+ pandas
+ scikit-image
+ scikit-learn 
+ seaborn


### For Cifar-10:
Bayesian fine-tuning:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_cifar.py --cutout 
```
  
  
### For ImageNet:
Bayesian fine-tuning:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_imagenet.py
```

### For face:
Bayesian fine-tuning:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_face.py
```
