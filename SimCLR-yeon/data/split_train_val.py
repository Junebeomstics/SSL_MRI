import pandas as pd
import os
import glob

filePath = '/home/yeon/anaconda3/envs/2022class/meta-learn/yAwareContrastiveLearning/data/'

targetPattern = r"*.csv"
fileAll = list(glob.glob(targetPattern))

for file in fileAll:
    rowsize = sum(1 for row in (open(filePath + file, encoding='UTF-8')))
    newsize = 1445
    times = 0
    for i in range(1, rowsize, newsize):
        times += 1   # 폴더 내 파일을 하나씩 점검하면서, 입력한 newsize보다 넘는 행을 쪼개줌
        df = pd.read_csv(filePath + file, header=None, nrows = newsize, skiprows=i)
        csv_output = file[:-4] + '_' + str(times) + '.csv'   # 쪼갠 수만큼 _1, _2... _n으로 꼬리를 달아서 파일명이 저장됨
        df.to_csv(filePath + csv_output, index=False, header=False, mode='a', chunksize=rowsize)
