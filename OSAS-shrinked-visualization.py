import pandas as pd
import numpy as np
import pickle
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

# Path to the dataset (assuming it is in the same directory as the notebook)
path = 'datasets/dataset_OSAS.feather'
pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

dataset = pd.read_feather(path)  
    
print(dataset.dtypes)
print("Number of rows:", dataset.shape[0])
print("Number of columns:", dataset.shape[1])
print("Number of distinct patients:", len(dataset['patient'].unique()))

# Function that, given an array of boolean values, outputs the "begin_index" and "end_index" of each contiguous block of TRUEs
def one_runs(a):
    iszero = np.concatenate(([0], np.equal(a, 1).astype(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges



# Calculate validation data for each patient
validation_pandas = []
pbar = tqdm(desc="Processed patients", total=len(dataset['patient'].unique()))
for pat in np.unique(dataset['patient'])[np.argsort(np.unique(dataset['patient']).astype(np.int8))]:
    temp = dataset[dataset['patient'] == pat]

    tmp_null_pleth = np.asarray([np.isnan(x) for x in temp['signal_pleth']]).flatten()
    
    pandas_row = [pat, # patient ID
                  round(len(temp) / 3600, 1), # recording duration (hours)
                  round(len(one_runs(temp['anomaly'].values)) / (len(temp) / 3600), 1), # AHI 
                  len(one_runs(temp[temp['event'] != 'HYPOPNEA']['anomaly'])), # number of apnea events
                  len(one_runs(temp[(temp['event'] != 'APNEA-CENTRAL') & (temp['event'] != 'APNEA-MIXED') & (temp['event'] != 'APNEA-OBSTRUCTIVE')]['anomaly'])), # number of hypopnea events
                  round(np.mean([x[1] - x[0] + 1 for x in one_runs(temp['anomaly'].values)])), # average duration of (hypo)apnea events (seconds)
                  round(np.std([x[1] - x[0] + 1 for x in one_runs(temp['anomaly'].values)])), # standard deviation of the duration of (hypo)apnea events (seconds)
                  round(100 * np.sum(np.isnan(temp['SpO2(%)'])) / len(temp), 1), # percentage of null SpO2 values
                  round(100 * np.sum(np.isnan(temp['PI(%)'])) / len(temp), 1), # percentage of null PI values
                  round(100 * np.sum(tmp_null_pleth) / len(tmp_null_pleth), 1), # percentage of null pleth values,
                 ]
    
    validation_pandas.append(pandas_row)
    
    pbar.update(1)
    
        
validation_pandas = pd.DataFrame(validation_pandas, columns=['patient', 
                                                             'recording duration (hrs)', 
                                                             'AHI', 
                                                             '# apnea events', 
                                                             '# hypopnea events', 
                                                             'avg duration (hypo)apnea events',
                                                             'stddev duration (hypo)apnea events',
                                                             '% null SpO2',
                                                             '% null PI', 
                                                             '% null pleth',
                                                            ])

print(validation_pandas)

# Boxplots of the ECG and PPG derived data
plt.figure(figsize=(20, 20))
pbar = tqdm(desc="Processed features", total=5)
for i, column in enumerate(['SpO2(%)', 'PI(%)']):
    plt.subplot(2, 1, i+1)
    plot_data = []
    for pat in np.unique(dataset['patient'])[np.argsort(np.unique(dataset['patient']).astype(np.int8))]:
        temp = dataset[dataset['patient'] == pat][column]
        plot_data.append([x for x in list(temp.values) if not np.isnan(x)])
    plt.boxplot(plot_data)
    if i == 4:
        plt.xlabel("Patient ID")
    plt.ylabel(column)
    pbar.update(1)
pbar.close()
plt.show()


# The following code re-assembles the time series related to each patient

patient_map_features = {} # given a patient, the map returns a map that, given feature, returns its whole time series
pbar = tqdm(desc="Processed patients", total=len(dataset['patient'].unique()))
for pat in dataset['patient'].unique():
    temp = dataset[dataset['patient'] == pat]
    feature_map_ts = {}
    for col in dataset.columns[1:]:
        if 'signal' not in col and 'PSG_' not in col:
            feature_map_ts[col] = temp[col].values
        else:
            feature_map_ts[col] = np.concatenate(temp[col].values)
    patient_map_features[pat] = feature_map_ts
    pbar.update(1)
pbar.close()


print(patient_map_features['1']['timestamp_datetime'][:5]) # for example, here are the first 5 timetamp values related to patient '1'
print(patient_map_features['1'])

# PI time series of patient 1. Observe that there are some outlier values that might be usefult correcting, for example using a weighting average approach.
plt.figure(figsize=(20, 6))
plt.plot(patient_map_features['1']['PI(%)'])
plt.show()


# Missing data in patient 10's SpO2 time series
plt.figure(figsize=(20, 6))
plt.plot(patient_map_features['10']['SpO2(%)'])
plt.show()

# Missing data in patient 10's SpO2 time series
plt.figure(figsize=(20, 6))
plt.plot(patient_map_features['10']['signal_pleth'])
plt.show()


