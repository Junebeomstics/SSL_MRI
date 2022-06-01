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


## ADNI Supervised Learning Test 결과
 **1. AD vs CN (N=100 str)**  
-![image](https://user-images.githubusercontent.com/64460370/171373516-667f11dd-f4e2-44af-ace5-fea17f7ef142.png)
-![ADNI_ADCN_strat_100_ROC](https://user-images.githubusercontent.com/64460370/171373908-bf23c7db-e162-4fd0-9d2c-6e48adbd4a26.png)
"/home/yeon/anaconda3/envs/2022class/meta-learn/SSL_MRI-wonyoung/SSL_MRI-wonyoung/ckpts/ADNI_ADCN_strat_100.pt"
- str: stratified / bal: balanced 
| **1. AD vs CN (N=100 str)** | Baseline | UKB_age |
| :---: | :---: | :---: |
| (freeze=F) | Test loss: 0.5045, Test accuracy: 74.38%, MEAN/AD/CN: 0.7766 |  | 
| (freeze=T) | ACC: 61.0%, AUC: 0.68 |  | 
| **2. AD vs CN (N=100 str)** |  |  |
| (freeze=F) | ACC: 81.2%, AUC: 0.86 |  | 
| (freeze=T) | ACC: 72.4%, AUC: 0.56 |  | 
