# SSL_MRI

Team members: 권준범, 장원영, 채연 

## Introduction

This repository is for Team project of 'Meta learning' class.
The main focus of our project is to observe the applicability of self-supervised learning to MRI images to predict clinical outcomes.

The main contributions of our project are as follows:
- Large scale pretraining with UK Biobank datasets with more than 40K samples.
- Based on y-aware contrastive learning module, we observe the effect of several metadata as an anchor to model the distance between negative pairs. (age, sex, GPS) 
- we verify the contribution of several transformation methods to make positive samples in self supervised learning.
- we compare the performances of famous baseline models such as Simclr, Moco, Model Genesis.
- we develop new framework that can integrate the self-supervised learning process with multi-task learning

Pre-training 모드 최종 코드 실행 예시는 아래와 같습니다. 각 arguement에 대한 설명은 `main.cv` 파일을 참고해주세요.
```bash
python main.py --mode pretraining --framework yaware --ckpt_dir ./checkpoint_yaware
```
