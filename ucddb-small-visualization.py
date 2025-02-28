import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt


# Path to the dataset (assuming it is in the same directory as the notebook)
path = 'datasets/dataset_UCDDB.feather'
events = 'datasets/dataset_UCDDB_events.feather'
pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199


dataset = pd.read_feather(path)  
events = pd.read_feather(events)


print(events.head())

print(dataset.head())



# Initialize a label column in the SpO2 dataframe
dataset['Label'] = 0  # Default label


# Get unique patients
patients = events['Patient'].unique()

# Iterate through each patient
for patient in patients:
    # Filter events and SpO2 data for the current patient
    patient_events = events[events['Patient'] == patient]
    patient_spo2 = dataset[dataset['Patient'] == patient]
    
    # Iterate through each event for the current patient
    for _, event in patient_events.iterrows():
        start_time = event['Time']
        end_time = start_time + event['Duration']
        event_type = 1
        
        # Find SpO2 measurements within the event's time range
        mask = (patient_spo2['Time'] >= start_time) & (patient_spo2['Time'] <= end_time)
        # Update the labels in the main dataset
        dataset.loc[patient_spo2[mask].index, 'Label'] = event_type

# Display the labeled dataframe

dataset.to_feather('datasets/dataset_UCDDB_labeled.feather')
print(dataset[dataset['Label'] == 1])