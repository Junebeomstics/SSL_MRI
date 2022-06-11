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



## ADNI Dataset Adaptation (220513 commit `6b5f03d`) - wonyoung

ADNI 데이터셋에 맞춘 Pretraining 모드 및 Finetuning 모드 코드를 추가했습니다.

`.py` 파일 내에 `# ADNI` (코드 라인), 혹은 `### ADNI` (코드 블록) 표시한 부분이 original 코드에서 수정한 부분입니다.

**아래는 이전 commit 버전이 아닌, 논문 original 코드와 비교한 내용입니다.**

- `main.py`
  - python argument 3개를 추가했습니다.
  - `--task_name`은 Finetuning 모드 task의 이름으로, `AD` 혹은 `MCI`를 입력합니다.
  - `--task_target_num`은 Finetuning 모드 task의 `N_train` 숫자로, `100` 혹은 `500`을 입력합니다.
  - `--stratify`는 Finetuning 모드 task의 `N_train` 숫자의 stratify 여부로, `strat` 혹은 `balan`을 입력합니다.
  - Test dataloader를 추가로 정의하고, AUROC를 산출하는 코드를 추가했습니다.
  - AUROC 은 `figs` 디렉토리에 저장됩니다.
  - 코드 실행 시간을 측정하기 위해 `time` 모듈을 가져왔습니다. 

- `dataset.py`
  - `MRIDataset` class명을 `ADNI_Dataset`로 변경했습니다.
  - `ADNI_Dataset` class가 `main.py` 실행 시 입력한 python argument 3개를 추가 argument로 받도록 수정했습니다. 이는 python argument에 맞는 데이터셋을 불러오기 위함입니다.
  - 데이터를 불러오는 코드를 ADNI 데이터셋에 맞게 수정했습니다. Pretraining 모드와 Finetuning 모드를 모두 고려해 수정했습니다.

- `datasplit.py`
  - `fsdat_baseline.csv` 파일을 Train, Valid, Test로 나누기 위해 새로 추가한 파일입니다.
  - Finetuning 모드 task로 1) AD vs CN, 2) MCI vs CN 두 가지 task를 정의하고 `.csv` 파일을 생성했습니다.
  - 각 task별로 `N_train = 100`과 `N_train = 500`을 나눠서 가정했습니다.
  - AD vs CN의 경우 데이터 불균형이 있어서 `N_train = 100`에 대해 stratification 적용 여부를 다시 나눴습니다.
  - 총 task의 개수는 5개입니다. 자세한 task 구성 정보 및 `.csv` 파일명은 `data_config.txt` 파일을 참고하세요.
  - Pretraining 모드를 위해 `CN_train.csv`와 `CN_valid.csv`를 생성했습니다.
  - `.csv` 파일은 편의상 `csv` 디렉토리에 정리했습니다.

- `yAwareContrastiveLearning.py`
  - Epoch마다 Training accuracy와 Validation accuracy를 확인할 수 있도록 수정했습니다.
  - 일정 epoch 동안 Validation loss에 개선이 없는 경우 early stopping을 실행하도록 수정했습니다.
  - Best 모델을 `ckpts` 디렉토리에 저장하도록 수정했습니다.
  - Test 데이터셋을 처리하는 코드를 추가했습니다.
  - Train, Valid, Test loss 계산식을 일부 수정했습니다.

- 기타
  - Early stopping을 위해 `Earlystopping.py`를 추가했습니다.
  - Finetuning 모드 실행 시 early stopping을 위해 `config.py`에 `self.patience=20`을 추가했습니다.
  - `.sh` 파일은 편의상 `shell` 디렉토리에 정리했습니다.
  - Train 및 Test 결과는 `shell` 디렉토리에 `.txt` 파일을 확인하세요.
  - 코드를 재현하려면 [여기](https://drive.google.com/file/d/1e75JYkaXvLQJhn0Km99iVTzB28AvErh5/view)에서 pretrained 모델을 다운 받으세요.

코드 실행 예시는 아래와 같습니다.
```bash
python3 main.py --mode finetuning --task_name AD --task_target_num 100 --stratify balan
```



## ADNI Dataset Adaptation (220520 commit `e2b229b`) - wonyoung

**아래는 commit `6b5f03d` 에서 수정한 내용입니다.**

- `main.py`
  - `--task_name`은 Finetuning 모드 task의 이름으로, AD vs MCI task를 추가함에 따라 `ADCN`, `MCICN`, `ADMCI` 중 하나를 입력합니다.
  - AUROC plot을 두 개 class 모두 생성하도록 코드를 수정했습니다.

- `dataset.py`
  - AD vs MCI task를 추가함에 따라 `__getitem__` method를 일부 수정했습니다.

- `datasplit.py`
  - AD vs MCI task를 추가해 이에 따른 `.csv` 파일을 생성했습니다.
  - AD vs MCI task도 `N_train = 100`과 `N_train = 500`을 나눠서 가정했습니다.
  - AD vs MCI task의 경우 데이터 불균형이 있어서 AD vs CN task처럼 `N_train = 100`에 대해 stratification 적용 여부를 다시 나눴습니다.
  - 이에 따라 총 task의 개수는 8개가 됐습니다. 자세한 task 구성 정보 및 `.csv` 파일명은 `data_config.txt` 파일을 참고하세요.

- `yAwareContrastiveLearning.py`
  - Representation에 대한 freeze 여부에 따라 다른 optimizer를 사용하도록 `self.optimizer` 부분을 수정했습니다.

- 기타
  - Representation에 대한 freeze 여부를 결정하기 위해 `config.py`에 `self.freeze`를 추가했습니다.
  - 추가 task의 Train 및 Test 결과는 `shell` 디렉토리에 `.txt` 파일을 확인하세요.

수정된 코드 실행 예시는 아래와 같습니다.
```bash
python3 main.py --mode finetuning --task_name ADCN --task_target_num 100 --stratify balan
```


## ADNI Dataset Adaptation (220527 commit `9a56b08`) - wonyoung

**아래는 commit `e2b229b` 에서 수정한 내용입니다. 주요 변경 사항은 다음과 같습니다.**  
**1. Pretraining 모드 실행 가능**  
**2. Multiple meta-data 입력 가능**  

- `main.py`
  - `main.py` 파일 내에서 train, valid, test 데이터셋을 생성하도록 코드를 추가했습니다.
  - 기존처럼 stratification 여부, training sample 수를 조절할 수 있습니다.
  - `--task_name`은 `--task_names`로 수정했습니다. 
  - `--task_names`는 Finetuning 모드 실행 시에만 argument를 입력하며, `AD/CN`, `MCI/CN`, `AD/MCI` 등과 같이 class label 2개를 `/`로 구분하여 입력합니다.
  - `--task_target_num`은 `--train_num`으로 수정했습니다.
  - `--train_num`은 어떤 모드로 실행해도 argument를 입력해야 합니다. Pretraining 모드에서도 training sample 수로 반영됩니다.
  - `--train_num`은 이제 100이나 500 등 고정된 숫자를 입력할 필요가 없습니다. 
  - 단, stratification 여부 등 일부 조건 등에 의해 코드 실행이 제한될 수 있습니다. 예를 들어 Test set이 100개 미만이면 코드가 실행되지 않습니다.
  - `ADNI` 데이터셋이 아닌 다른 데이터셋을 써도 dataset 부분만 수정하면 실행 가능하도록 (최대한) 코드를 수정했습니다.

- `dataset.py`
  - `main.py` 파일 내에서 train, valid, test 데이터셋을 생성함에 따라 지정된 데이터를 가져오는 역할만 하도록 간소화했습니다.

- `datasplit.py`
  - `main.py` 파일 내에서 train, valid, test 데이터셋을 생성함에 따라 Pretraining 모드를 위한 데이터셋만 따로 생성하도록 수정했습니다.

- `losses.py`
  - `GeneralizedSupervisedNTXenLoss`가 multiple meta-data를 고려해 계산될 수 있도록 코드를 수정했습니다.
  - 각종 하이퍼파라미터를 받아오기 위해 loss 선언 시 `config` 파일을 input으로 추가했습니다.
  - Multiple meta-data를 고려함에 따라 `sigma`, `alpha_list` 등도 list 객체로 받아옵니다.

- `config.py`
  - Pretraining 모드 관련 객체를 추가했습니다.
  - `self.label_name`은 각 meta-data의 변수명을 담은 list입니다.
  - `self.label_type`은 각 meta-data의 type입니다. 변수가 continuous라면 `cont`, catrgorical이면 `cat`으로 입력합니다.
  - `self.alpha_list`는 각 meta-data의 weight입니다. 총합은 1이 돼야 합니다.
  - `self.sigma`는 각 meta-data의 sigma parameter입니다.

수정된 코드 실행 예시는 아래와 같습니다.
```bash
python3 main.py --mode finetuning --task_names AD/CN --train_num 100 --stratify balan
python3 main.py --mode pretraining --train_num 100
```



## ADNI Dataset Adaptation (220603 commit `ebba070`) - wonyoung

**아래는 commit `9a56b08` 에서 수정한 내용입니다. 주요 변경 사항은 다음과 같습니다.**  
**1. Finetuning 모드에서 regression task 실행 가능**  
**2. Finetuning 모드에서 layer별로 lr 다르게 적용 가능**  
**3. Pretraining 모드에서 RBF 커널에 기반한 categorical loss 구현**  
**4. Reproducibility 고려 가능**  

- `main.py`
  - Finetuning 모드에서 regression task를 실행할 수 있도록 수정했습니다.
  - Regression task도 training sample 수를 조절할 수 있지만, stratification 여부는 고려하지 않았습니다.
  - Regression task의 loss는 `MSELoss`를 쓰도록 설정했습니다.
  - Regression task는 `MSE`, `MAE`, `RMSE` 등 metric을 산출합니다.
  - `--task_names`를 `--task_name`로 수정했습니다.
  - `--random_seed`를 argument로 지정해 실험을 재현할 수 있습니다.

- `config.py`
  - Finetuning 모드에서 task 종류에 따라 `task_type`을 설정하도록 추가했습니다. 분류 task에는 `cls`를, 회귀 task에는 `reg`을 입력합니다.
  - `self.num_classes`는 classification task에는 `2`를, regression task에는 `1`을 입력합니다.
  - Finetuning 모드에서 `self.freeze`를 `self.layer_control`로 변경했습니다. Pretraining 모드에서는 해당 옵션을 삭제했습니다.
  
- `yAwareContrastiveLearning.py`
  - Finetuning 모드에서 `task_type`에 따라 데이터 타입을 조정하기 위해 조건문을 일부 추가했습니다.
  - Finetuning 모드에서 `self.layer_control`에 따라 layer별로 learning rate를 다르게 적용할 수 있습니다.

- 기타
  - `losses.py`에 Pretraining 모드에서 동작하는 RBF 커널 기반 categorical loss를 구현했습니다.
  - `.pth` Pretraining 파일들을 모아놓은 `weights` 디렉토리를 추가했습니다.

수정된 코드 실행 예시는 아래와 같습니다.
```bash
python3 main.py --mode pretraining --train_num 100 --random_seed 0
python3 main.py --mode finetuning --train_num 100 --task_name PTAGE --random_seed 0
python3 main.py --mode finetuning --train_num 100 --task_name AD/CN --stratify balan --random_seed 0
```



## ADNI Dataset Adaptation (220611 commit `b84bc18`) - wonyoung

**아래는 commit `ebba070` 에서 수정한 내용입니다. 주요 변경 사항은 다음과 같습니다.**  
**1. Pretraining 모드에서 categorical loss를 hyperparameter화**  
**2. 5-fold stratified CV Leave-Site-Out 구현 및 실험 중**  

- `main.py`
  - `--layer_control`를 추가해 기존에 `config.py`에서 제어하던 `self.layer_control`을 argument로 주도록 했습니다.
  - `--random_seed`는 random augmentation에는 영향을 주지 않습니다.

- `dataset.py`
  - Finetuning 모드에서 train 데이터셋만 augmentation을 적용하도록 수정했습니다.

- `losses.py`
  - `self.config.cat_similarity`로 `config.py`에서 categorical meta-data의 similarity를 주도록 했습니다.

- `main_cv.py`
  - Finetuning 모드만을 위한 파일입니다.
  - 모든 task에 대해 balanced sample을 세팅합니다.
  - `--layer_control` argument로 `self.layer_control`을 조절합니다.
  - 5-fold stratified CV Leave-Site-Out을 구현했습니다.
 
수정된 코드 실행 예시는 아래와 같습니다.
```bash
python3 main.py --mode pretraining --train_num 100 --random_seed 0
python3 main.py --mode finetuning --train_num 100 --task_name PTAGE --layer_control tune_all --random_seed 0
python3 main.py --mode finetuning --train_num 100 --task_name AD/CN --layer_control freeze --stratify balan --random_seed 0

python3 main_cv.py --train_num 100 --task_name AD/CN --layer_control tune_all --random_seed 0
python3 main_cv.py --train_num 300 --task_name AD/CN --layer_control freeze --random_seed 0
```


## ADNI Finetuning 모드 Test 결과
- Samples are balanced for all tasks.
- Average AUC for 5-fold stratified CV Leave-Site-Out  
- [0:3] : Pretraining dataset (BHB / UKB)
- [3] : Pretraining model type (yAware / SimCLR)
- [4] : Pretraining meta-data (age / sex / intelligence GPS)
- [5:7] : Pretraining batch size (32 / 64)
- [7] : Pretraining augmentation type (all_tf / cutout)

| **1. AD vs CN (N=100)** | BHBya64c | UKBya32a | UKBsa32a | UKBya64a | UKBsa64a |
| :---: | :---: | :---: | :---: | :---: | :---: |
| (tune)   | 0.83 |  |  |  |  |
| (freeze) | 0.71 |  |  |  |  |
| **2. AD vs CN (N=300)** |  |  |  |  |  |
| (tune)   | 0.90 |  |  |  |  |
| (freeze) | 0.76 |  |  |  |  |
| **3. AD vs MCI (N=100)** |  |  |  |  |  |
| (tune)   | 0.69 |  |  |  |  |
| (freeze) | 0.65 |  |  |  |  |
| **4. AD vs MCI (N=300)** |  |  |  |  |  |
| (tune)   | 0.77 |  |  |  |  |
| (freeze) | 0.68 |  |  |  |  |
| **5. MCI vs CN (N=100)** |  |  |  |  |  |
| (tune)   | 0.59 |  |  |  |  |
| (freeze) | 0.57 |  |  |  |  |
| **6. MCI vs CN (N=500)** |  |  |  |  |  |
| (tune)   | 0.57 |  |  |  |  |
| (freeze) | 0.58 |  |  |  |  |

| **1. AD vs CN (N=100)** | UKBya64c | UKBsa64c | UKBys64c | UKByi64c | UKBsa64c+UKBya64c |
| :---: | :---: | :---: | :---: | :---: | :---: |
| (tune)   |  |  |  |  |  |
| (freeze) |  |  |  |  |  |
| **2. AD vs CN (N=300)** |  |  |  |  |  |
| (tune)   |  |  |  |  |  |
| (freeze) |  |  |  |  |  |
| **3. AD vs MCI (N=100)** |  |  |  |  |  |
| (tune)   |  |  |  |  |  |
| (freeze) |  |  |  |  |  |
| **4. AD vs MCI (N=300)** |  |  |  |  |  |
| (tune)   |  |  |  |  |  |
| (freeze) |  |  |  |  |  |
| **5. MCI vs CN (N=100)** |  |  |  |  |  |
| (tune)   |  |  |  |  |  |
| (freeze) |  |  |  |  |  |
| **6. MCI vs CN (N=500)** |  |  |  |  |  |
| (tune)   |  |  |  |  |  |
| (freeze) |  |  |  |  |  |



## 주간과제
- [x] ADNI Finetuning 모드 실험하기 (5 tasks)
- [x] AD vs MCI 추가 실험하기 (3 tasks)
- [x] Representation freeze 하고 실험하기 (8 tasks)
- [x] ADNI Pretraining 모드 구현하기
- [x] dataset.py 등 프레임워크 개선하기
- [x] Multiple meta-data 프레임워크 구현하기
- [x] Finetuning 모드에서 regression task 구현하기
- [x] Reproducibility 구현하기
- [x] UKB pretrained weight로 ADNI Finetuning 모드 실험하기
- [x] Categorical loss kernel 구현하기
- [x] Finetuning 모드에서 layer별로 lr 다르게 적용하기
- [x] Categorical loss constant term configuration으로 넣기
- [x] Finetuning 모드 stratification 모드 버리기 (N, freeze 여부는 유지) (12 tasks)
- [x] Finetuning 모드 task 별로 학습 및 평가 자동화하기
- [x] Train set label 비율 1대1로 고정하고 5-fold stratified CV 실행하기 (각 N=100, N=300, N=500)
