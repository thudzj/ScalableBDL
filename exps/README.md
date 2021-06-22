## Refer to [here](https://github.com/thudzj/BayesAdapter/tree/prac) for the implementation for [BayesAdapter: Being Bayesian, Inexpensively and Reliably, via Bayesian Fine-tuning](https://arxiv.org/pdf/2010.01979.pdf).

## Reproduce the results in [LiBRe: A Practical Bayesian Approach to Adversarial Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_LiBRe_A_Practical_Bayesian_Approach_to_Adversarial_Detection_CVPR_2021_paper.pdf)


At first, execute `pip install git+https://github.com/thudzj/ScalableBDL.git@efficient`

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


### For ImageNet:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_imagenet.py --job-id emp-nores-blur.03-6epoch-mineps.5-2 --posterior_type emp --blur_prob 0.03 --epochs 6 --epsilon_min_train 0.5
```

### For face:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_face.py --job-id face_ir50_softmax-alpha100 --posterior_type emp --data_path /data/zhijie/faces_emore --epsilon_min_train 1 --epsilon_max_train 2 --blur_prob 0 --uncertainty_threshold 1 --arch IR_50 --head Softmax --alpha 100
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_face.py --job-id face_ir50_cos-alpha100 --posterior_type emp --data_path /data/zhijie/faces_emore --epsilon_min_train 1 --epsilon_max_train 2 --blur_prob 0 --uncertainty_threshold 1 --arch IR_50 --head CosFace --alpha 100
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_face.py --job-id face_ir50_arc-alpha100 --posterior_type emp --data_path /data/zhijie/faces_emore --epsilon_min_train 1 --epsilon_max_train 2 --blur_prob 0 --uncertainty_threshold 1 --arch IR_50 --head ArcFace --alpha 100
```
