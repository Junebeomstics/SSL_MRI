import pandas as pd
import glob
import shutil

filePath = '/home/yeon/anaconda3/envs/2022class/meta-learn/yAwareContrastiveLearning/data/'

targetPattern = r"*_val.csv"
file = glob.glob(targetPattern)

df = pd.read_csv(filePath + file[0])
file_list = list(df['File_name'])
for f in file_list:
    target_file = '/home/yeon/anaconda3/envs/2022class/meta-learn/yAwareContrastiveLearning/data/adni_t1s_baseline/' + f
    moved = '/home/yeon/anaconda3/envs/2022class/meta-learn/yAwareContrastiveLearning/data/val/' + f
    shutil.move(target_file, moved)