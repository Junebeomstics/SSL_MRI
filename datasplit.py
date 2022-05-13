import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('./csv/fsdat_baseline.csv')
data['Dx.new'].value_counts()

# AD (281) vs CN (738)
data_ADCN = data[data['Dx.new'] != 'MCI']
y_ADCN = data_ADCN['Dx.new']

## For N_train = 100
### Stratified sample ratio
ADCN_tv100, ADCN_test100, _, _ = train_test_split(data_ADCN, y_ADCN, test_size=894, shuffle=True, stratify=y_ADCN, random_state=34)
y_ADCN_tv100 = ADCN_tv100['Dx.new']
ADCN_train100, ADCN_valid100, _, _ = train_test_split(ADCN_tv100, y_ADCN_tv100, test_size=25, shuffle=True, stratify=y_ADCN_tv100, random_state=34)
ADCN_train100.to_csv('./csv/ADCN_strat_train100.csv', index = False)
ADCN_valid100.to_csv('./csv/ADCN_strat_valid100.csv', index = False)
ADCN_test100.to_csv('./csv/ADCN_strat_test100.csv', index = False)

## For N_train = 100
### Balanced sample ratio
data_AD = data[data['Dx.new'] == 'AD']
data_CN = data[data['Dx.new'] == 'CN']

ADCNb_train100 = pd.concat([data_AD[:50], data_CN[:50]])
ADCNb_valid100 = pd.concat([data_AD[50:62], data_CN[50:63]])
ADCNb_test100 = pd.concat([data_AD[62:281], data_CN[63:282]])
ADCNb_train100.to_csv('./csv/ADCN_balan_train100.csv', index = False)
ADCNb_valid100.to_csv('./csv/ADCN_balan_valid100.csv', index = False)
ADCNb_test100.to_csv('./csv/ADCN_balan_test100.csv', index = False)

## For N_train = 500
### Stratified sample ratio
ADCN_tv500, ADCN_test500, _, _ = train_test_split(data_ADCN, y_ADCN, test_size=419, shuffle=True, stratify=y_ADCN, random_state=34)
y_ADCN_tv500 = ADCN_tv500['Dx.new']
ADCN_train500, ADCN_valid500, _, _ = train_test_split(ADCN_tv500, y_ADCN_tv500, test_size=100, shuffle=True, stratify=y_ADCN_tv500, random_state=34)
ADCN_train500.to_csv('./csv/ADCN_strat_train500.csv', index = False)
ADCN_valid500.to_csv('./csv/ADCN_strat_valid500.csv', index = False)
ADCN_test500.to_csv('./csv/ADCN_strat_test500.csv', index = False)



# MCI (788) vs CN (738)
data_MCICN = data[data['Dx.new'] != 'AD']
y_MCICN = data_MCICN['Dx.new']

## For N_train = 100
### Stratified/balanced sample ratio
MCICN_tv100, MCICN_test100, _, _ = train_test_split(data_MCICN, y_MCICN, test_size=1401, shuffle=True, stratify=y_MCICN, random_state=34)
y_MCICN_tv100 = MCICN_tv100['Dx.new']
MCICN_train100, MCICN_valid100, _, _ = train_test_split(MCICN_tv100, y_MCICN_tv100, test_size=25, shuffle=True, stratify=y_MCICN_tv100, random_state=34)
MCICN_train100.to_csv('./csv/MCICN_strat_train100.csv', index = False)
MCICN_valid100.to_csv('./csv/MCICN_strat_valid100.csv', index = False)
MCICN_test100.to_csv('./csv/MCICN_strat_test100.csv', index = False)

## For N_train = 500
### Stratified/balanced sample ratio
MCICN_tv500, MCICN_test500, _, _ = train_test_split(data_MCICN, y_MCICN, test_size=926, shuffle=True, stratify=y_MCICN, random_state=34)
y_MCICN_tv500 = MCICN_tv500['Dx.new']
MCICN_train500, MCICN_valid500, _, _ = train_test_split(MCICN_tv500, y_MCICN_tv500, test_size=100, shuffle=True, stratify=y_MCICN_tv500, random_state=34)
MCICN_train500.to_csv('./csv/MCICN_strat_train500.csv', index = False)
MCICN_valid500.to_csv('./csv/MCICN_strat_valid500.csv', index = False)
MCICN_test500.to_csv('./csv/MCICN_strat_test500.csv', index = False)
