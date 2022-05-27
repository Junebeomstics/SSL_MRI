import pandas as pd

# Pre-training Dataset Generation (ADNI CN)
data = pd.read_csv('./csv/fsdat_baseline.csv')
data_CN = data[data['Dx.new'] == 'CN']
data_CN.to_csv('./csv/fsdat_baseline_CN.csv', index = False)
