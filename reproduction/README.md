# BayesAdapter

For Cifar-10:

Deterministic pre-training:
  >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --cutout --job-id map-decay2 --batch_size 256 --decay 0.0002
  
Bayesian fine-tuning:
  >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune.py --cutout  --job-id ft-gan1000-.75 --bayes mf --epochs 40 --schedule  10 20 30 --num_gan 1000 --mi_th 0.75 --dist-port 2345
  
  
For ImageNet:

Download the deterministic checkpoint of ResNet-50 trained on ImageNet from https://download.pytorch.org/models/resnet50-19c8e357.pth

Bayesian fine-tuning:
  >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_in.py --job-id ft-gan-.75-1-alpha3 --alpha 3 --bayes mf --epochs 12 --schedule 3 6 9 --log_sigma_init_range -6 -5 --num_gan 1000 --mi_th 0.75 --epsilon_scale 1



For face:

Deterministic pre-training:
  >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_face.py  --job-id map-ft0.01-drop0-decay5 --dist-port 2345 --ft_learning_rate 0.01 --epochs 90 --schedule 30 60 80 --dropout_rate 0 --decay 5e-4
  
Bayesian fine-tuning:
  >> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_face.py --job-id ft-gan-.75-1-epo16-d1 --bayes mf --epochs 16 --schedule 4 8 12 --log_sigma_init_range -6 -5 --num_gan 1000 --mi_th 0.75 --decay 5e-4 --fc_bayes mf --epsilon_scale 0.25
