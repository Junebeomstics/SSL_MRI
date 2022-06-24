# SSL_MRI

Team members: 권준범, 류명훈, 장원영, 채연 

## Introduction

This repository is for Team project of 'Meta learning' class.
The main focus of our project is to observe the applicability of self-supervised learning to MRI images to predict clinical outcomes.

The main contributions of our project are as follows:
- Large scale pretraining with UK Biobank datasets with more than 40K samples.
- Based on y-aware contrastive learning module, we observe the effect of several metadata as an anchor to model the distance between negative pairs. (age, sex, GPS) 
- we verify the contribution of several transformation methods to make positive samples in self supervised learning.
- we compare the performances of famous baseline models such as Simclr, Moco, Model Genesis.
- we develop new framework that can integrate the self-supervised learning process with multi-task learning

## Pre-training 모드에 대한 구현 및 코드 실행은 `pretraining` 브랜치에서 확인하실 수 있습니다.
## Fine-tuning 모드에 대한 구현 및 코드 실행은 `Finetuning` 브랜치에서 확인하실 수 있습니다.

감사합니다.
