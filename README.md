# auxSKD

Pytorch implementation of [Auxiliary Learning for Self-Supervised Video Representation via Similarity-based Knowledge Distillation](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Dadashzadeh_Auxiliary_Learning_for_Self-Supervised_Video_Representation_via_Similarity-Based_Knowledge_Distillation_CVPRW_2022_paper.pdf), published as a CVPR 2022 workshop paper. 


# Pretraining
## Data preparation
Follow instructions in [VideoPace](https://github.com/laura-wang/video-pace#data-preparation)

## Auxiliary Pretraining
cd auxSKD
python train.py  --gpu 0,1 --bs 30 --lr 0.001 --height 128 --width 171 --crop_sz 112 --clip_len 16

## Primary Pretraining - VSPP
cd ..__
cd VSPP__
python train.py  --gpu 0,1 --bs 30 --lr 0.001 --height 128 --width 171 --crop_sz 112 --clip_len 16


# Acknowlegement


Part of our codes are adapted from [ISD](https://github.com/UMBCvision/ISD) and [VideoPace](https://github.com/laura-wang/video-pace), we thank the authors for their contributions.
