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


### Fine-tuning and test for ADNI dataset - wonyoung

ADNI 데이터셋에 맞춘 fine-tuning code를 추가했습니다.
- 파이썬 파일 내에 `# ADNI` (코드 라인), 혹은 `### ADNI` (코드 블록) 표시가 된 부분이 원래 repository 파일에서 수정한 부분입니다.
- `main.py` 내에 test 데이터셋을 따로 정의하고, AUROC를 산출하는 코드를 추가했습니다.
- `dataset.py`를 ADNI 데이터에 맞게 수정했습니다.
- `datasplit.py`는 `fsdat_baseline.csv` 파일을 train, valid, test로 나누기 위해 새로 추가한 파일입니다.
- 논문 실험 세팅에 맞춰 우선 `N_train = 100 or 500`을 상정하고 `.csv` 파일을 생성했습니다.
- `N_train = 100 or 500`은 `main.py`를 실행할 때 python argument로 추가하도록 설정했습니다.
- `.csv` 파일은 편의상 `csv` 폴더에 정리했습니다.
- `yAwareContrastiveLearning.py` 내에 test 데이터셋을 처리하는 코드를 추가했습니다.
- 코드를 재현하려면 [여기](https://drive.google.com/file/d/1e75JYkaXvLQJhn0Km99iVTzB28AvErh5/view)에서 pretrained 모델을 다운 받으세요.
- '.sh' 파일은 편의상 `bash` 폴더에 정리했습니다.

코드 실행 예시는 아래와 같습니다.
```bash
python3 main.py --mode finetuning --target_num 100
```

* 학습 종료 후 AUROC plot 추가하겠습니다.
