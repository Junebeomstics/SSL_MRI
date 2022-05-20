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



## ADNI Dataset Adaptation (220513) - wonyoung

ADNI 데이터셋에 맞춘 Pre-training 및 Fine-tuning 코드를 추가했습니다.

`.py` 파일 내에 `# ADNI` (코드 라인), 혹은 `### ADNI` (코드 블록) 표시한 부분이 original 코드에서 수정한 부분입니다.

**아래는 이전 commit 버전이 아닌, 논문 original 코드와 비교한 내용입니다.**

- `main.py`
  - python argument 3개를 추가했습니다.
  - `--task_name`은 Fine-tuning task의 이름으로, `ADCN`, `MCICN`, `ADMCI` 중 하나를 입력합니다.
  - `--task_target_num`은 Fine-tuning task의 `N_train` 숫자로, `100` 혹은 `500`을 입력합니다.
  - `--stratify`는 Fine-tuning task의 `N_train` 숫자의 stratify 여부로, `strat` 혹은 `balan`을 입력합니다.
  - Test dataloader를 추가로 정의하고, AUROC plot을 산출하는 코드를 추가했습니다.
  - AUROC plot은 `figs` 디렉토리에 저장됩니다.
  - 코드 실행 시간을 측정하기 위해 `time` 모듈을 가져왔습니다. 

- `dataset.py`
  - `MRIDataset` class명을 `ADNI_Dataset`로 변경했습니다.
  - `ADNI_Dataset` class가 `main.py` 실행 시 입력한 python argument 3개를 추가 argument로 받도록 수정했습니다. 이는 python argument에 맞는 데이터셋을 불러오기 위함입니다.
  - 데이터를 불러오는 코드를 ADNI 데이터셋에 맞게 수정했습니다. Pre-training과 Fine-tuning을 모두 고려해 수정했습니다.

- `datasplit.py`
  - `fsdat_baseline.csv` 파일을 Train, Valid, Test로 나누기 위해 새로 추가한 파일입니다.
  - Fine-tuning task로 1) AD vs CN, 2) MCI vs CN, 3) AD vs MCI 세 가지 task를 정의하고 `.csv` 파일을 생성했습니다.
  - 각 task별로 `N_train = 100`과 `N_train = 500`을 나눠서 가정했습니다.
  - AD vs CN의 경우 데이터 불균형이 있어서 `N_train = 100`에 대해 stratification 적용 여부를 다시 나눴습니다.
  - AD vs MCI의 경우 데이터 불균형이 있어서 `N_train = 100`에 대해 stratification 적용 여부를 다시 나눴습니다.
  - 총 task의 개수는 8개입니다. 자세한 task 구성 정보 및 `.csv` 파일명은 `data_config.txt` 파일을 참고하세요.
  - Pre-training을 위해 `CN_train.csv`와 `CN_valid.csv`를 생성했습니다.
  - `.csv` 파일은 편의상 `csv` 디렉토리에 정리했습니다.

- `yAwareContrastiveLearning.py`
  - Epoch마다 Training accuracy와 Validation accuracy를 확인할 수 있도록 수정했습니다.
  - 일정 epoch 동안 Validation loss에 개선이 없는 경우 early stopping을 실행하도록 수정했습니다.
  - Best 모델을 `ckpts` 디렉토리에 저장하도록 수정했습니다.
  - Test 데이터셋을 처리하는 코드를 추가했습니다.
  - Train, Valid, Test loss 계산식을 일부 수정했습니다.

- 기타
  - Early stopping을 위해 `Earlystopping.py`를 추가했습니다.
  - Fine-tuning 실행 시 early stopping을 위해 `config.py`에 `self.patience=20`을 추가했습니다.
  - `.sh` 파일은 편의상 `shell` 디렉토리에 정리했습니다.
  - Train 및 Test 결과는 `shell` 디렉토리에 `.txt` 파일을 확인하세요.
  - 코드를 재현하려면 [여기](https://drive.google.com/file/d/1e75JYkaXvLQJhn0Km99iVTzB28AvErh5/view)에서 pretrained 모델을 다운 받으세요.

코드 실행 예시는 아래와 같습니다.
```bash
python3 main.py --mode finetuning --task_name ADCN --task_target_num 100 --stratify balan
```

## ADNI Fine-tuning 실험 결과
1. AD vs CN (N=100 stratified)
Test ACC: 81.21%, Test AUROC: 0.8606

2. AD vs CN (N=100 balanced)
Test ACC: 69.63%, Test AUROC: 0.8009

3. AD vs CN (N=500 stratified)
Test ACC: 83.05%, Test AUROC: 0.8937
 
4. MCI vs CN (N=100 stratified/balanced)
Test ACC: 53.03%, Test AUROC: 0.6209

5. MCI vs CN (N=500 stratified/balanced)
Test ACC: 61.12%, Test AUROC: 0.6445

6. AD vs MCI (N=100 stratified)
Test ACC: 73.31%, Test AUROC: 0.6590

7. AD vs MCI (N=100 balanced)
Test ACC: 56.62%, Test AUROC: 0.5862

8. AD vs MCI (N=500 stratified)
Test ACC: 75.05%, Test AUROC: 0.7505


## 주간과제
- [x] 학습 종료 후 AUROC plot 및 학습 결과 `.txt` 파일 추가하기
- [x] AD vs MCI 후 AUROC plot 및 학습 결과 `.txt` 파일 추가하기
- [ ] Backbone freeze 하고 학습 진행하기 (8 tasks)
- [ ] ADNI 데이터셋으로 Pre-training 진행하기
- [ ] multiple meta-data 활용 방안 확인하기
