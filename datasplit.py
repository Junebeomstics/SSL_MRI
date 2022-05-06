import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('./csv/fsdat_baseline.csv')
y = data['Dx.new']

# N_train = 100
# testset generation
data100, test100, _, _ = train_test_split(data, y, test_size=1703, shuffle=True, stratify=y, random_state=34)
# 5-fold CV generation
y1_100 = data100['Dx.new']
train1_100, valid1_100, _, _ = train_test_split(data100, y1_100, test_size=20, shuffle=True, stratify=y1_100, random_state=34)
y2_100 = train1_100['Dx.new']
train2_100, valid2_100, _, _ = train_test_split(train1_100, y2_100, test_size=20, shuffle=True, stratify=y2_100, random_state=34)
y3_100 = train2_100['Dx.new']
train3_100, valid3_100, _, _ = train_test_split(train2_100, y3_100, test_size=20, shuffle=True, stratify=y3_100, random_state=34)
y4_100 = train3_100['Dx.new']
train4_100, valid4_100, _, _ = train_test_split(train3_100, y4_100, test_size=20, shuffle=True, stratify=y4_100, random_state=34)
y5_100 = train4_100['Dx.new']
train5_100, valid5_100, _, _ = train_test_split(train4_100, y5_100, test_size=20, shuffle=True, stratify=y5_100, random_state=34)

# temp test file
datatemp, testtemp, _, _ = train_test_split(data, y, test_size=100, shuffle=True, stratify=y, random_state=34)
testtemp.to_csv('./csv/fsdat_baseline_test_temp.csv', index = False)

test100.to_csv('./csv/fsdat_baseline_test_100.csv', index = False)
valid1_100.to_csv('./csv/fsdat_baseline_valid1_100.csv', index = False)
valid2_100.to_csv('./csv/fsdat_baseline_valid2_100.csv', index = False)
valid3_100.to_csv('./csv/fsdat_baseline_valid3_100.csv', index = False)
valid4_100.to_csv('./csv/fsdat_baseline_valid4_100.csv', index = False)
valid5_100.to_csv('./csv/fsdat_baseline_valid5_100.csv', index = False)
pd.concat([valid2_100, valid3_100, valid4_100, valid5_100]).to_csv('./csv/fsdat_baseline_train1_100.csv', index = False)
pd.concat([valid1_100, valid3_100, valid4_100, valid5_100]).to_csv('./csv/fsdat_baseline_train2_100.csv', index = False)
pd.concat([valid1_100, valid2_100, valid4_100, valid5_100]).to_csv('./csv/fsdat_baseline_train3_100.csv', index = False)
pd.concat([valid1_100, valid2_100, valid3_100, valid5_100]).to_csv('./csv/fsdat_baseline_train4_100.csv', index = False)
pd.concat([valid1_100, valid2_100, valid3_100, valid4_100]).to_csv('./csv/fsdat_baseline_train5_100.csv', index = False)
pd.concat([valid1_100, valid2_100, valid3_100, valid4_100, valid5_100]).to_csv('./csv/fsdat_baseline_train_100.csv', index = False)



# N_train = 100
# testset generation
data500, test500, _, _ = train_test_split(data, y, test_size=1303, shuffle=True, stratify=y, random_state=34)
# 5-fold CV generation
y1_500 = data500['Dx.new']
train1_500, valid1_500, _, _ = train_test_split(data500, y1_500, test_size=100, shuffle=True, stratify=y1_500, random_state=34)
y2_500 = train1_500['Dx.new']
train2_500, valid2_500, _, _ = train_test_split(train1_500, y2_500, test_size=100, shuffle=True, stratify=y2_500, random_state=34)
y3_500 = train2_500['Dx.new']
train3_500, valid3_500, _, _ = train_test_split(train2_500, y3_500, test_size=100, shuffle=True, stratify=y3_500, random_state=34)
y4_500 = train3_500['Dx.new']
train4_500, valid4_500, _, _ = train_test_split(train3_500, y4_500, test_size=100, shuffle=True, stratify=y4_500, random_state=34)
y5_500 = train4_500['Dx.new']
train5_500, valid5_500, _, _ = train_test_split(train4_500, y5_500, test_size=100, shuffle=True, stratify=y5_500, random_state=34)

test500.to_csv('./csv/fsdat_baseline_test_500.csv', index = False)
valid1_500.to_csv('./csv/fsdat_baseline_valid1_500.csv', index = False)
valid2_500.to_csv('./csv/fsdat_baseline_valid2_500.csv', index = False)
valid3_500.to_csv('./csv/fsdat_baseline_valid3_500.csv', index = False)
valid4_500.to_csv('./csv/fsdat_baseline_valid4_500.csv', index = False)
valid5_500.to_csv('./csv/fsdat_baseline_valid5_500.csv', index = False)
pd.concat([valid2_500, valid3_500, valid4_500, valid5_500]).to_csv('./csv/fsdat_baseline_train1_500.csv', index = False)
pd.concat([valid1_500, valid3_500, valid4_500, valid5_500]).to_csv('./csv/fsdat_baseline_train2_500.csv', index = False)
pd.concat([valid1_500, valid2_500, valid4_500, valid5_500]).to_csv('./csv/fsdat_baseline_train3_500.csv', index = False)
pd.concat([valid1_500, valid2_500, valid3_500, valid5_500]).to_csv('./csv/fsdat_baseline_train4_500.csv', index = False)
pd.concat([valid1_500, valid2_500, valid3_500, valid4_500]).to_csv('./csv/fsdat_baseline_train5_500.csv', index = False)
pd.concat([valid1_500, valid2_500, valid3_500, valid4_500, valid5_500]).to_csv('./csv/fsdat_baseline_train_500.csv', index = False)
