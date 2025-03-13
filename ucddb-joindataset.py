import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt



# Constants for the segment splits
FS = 128  # Original sampling rate (Hz)
TARGET_FREQUENCY = 1  # Desired sampling frequency (Hz)
WINDOW_SIZE = int(FS * 60 / (FS // TARGET_FREQUENCY))  # Adjusted window size
HALF_WINDOW = WINDOW_SIZE // 2  # 50% overlap
APNEA_RATIO = 0.8  # Desired balance between apnea and non-apnea events
SAMPLE_INTERVAL = FS // TARGET_FREQUENCY  # How many original samples to skip

spo2_min_threshold = 68  # Not sure what to use here
apnea_types = ['APNEA-O', 'APNEA-C', 'APNEA-M']
hypopnea_types = ['HYP-O', 'HYP-C', 'HYP-M']
other_types = ['PB', 'POSSIBLE']
# "ucddb008", "ucddb011", "ucddb013" , "ucddb017", "ucddb018"
#patient split for train/test
train_patients = [
    "ucddb002", "ucddb003", "ucddb006", "ucddb007", "ucddb009",
    "ucddb010", "ucddb012", "ucddb019", "ucddb020", "ucddb022",
    "ucddb023", "ucddb025", "ucddb027", "ucddb028"
]

test_patients = [
    "ucddb005", "ucddb014", "ucddb015", "ucddb021", "ucddb024",
    "ucddb026"
]

def one_hot_encode_labels(df):
    """
    Adds one-hot encoded labels for Apnea, Hypopnea, and Other categories.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'Type' column.
    
    Returns:
        pd.DataFrame: Updated DataFrame with new one-hot encoded columns.
    """
    df = df.copy()  # Avoid modifying the original dataset
    
    # Define category groups
    apnea_types = ['APNEA-O', 'APNEA-C', 'APNEA-M']
    hypopnea_types = ['HYP-O', 'HYP-C', 'HYP-M']
    other_types = ['PB', 'POSSIBLE']

    # One-hot encode labels
    df['Apnea'] = df['Type'].apply(lambda x: 1 if x in apnea_types else 0)
    df['Hypopnea'] = df['Type'].apply(lambda x: 1 if x in hypopnea_types else 0)
    df['Other'] = df['Type'].apply(lambda x: 1 if x in other_types else 0)

    return df


path = 'datasets/dataset_UCDDB.feather'
events = 'datasets/dataset_UCDDB_events.feather'
pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199


dataset = pd.read_feather(path)  
events = pd.read_feather(events)


print(events.head())
print(events['Type'].unique())
#print(dataset.head()


# Initialize a label column in the SpO2 dataframe
dataset['Label'] = 0  # Default label
# Display the labeled dataframe
dataset = dataset[dataset["SpO2"] >= spo2_min_threshold].reset_index(drop=True)

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
        apnea_type = event['Type']
        event_type = 1
        
        # Find SpO2 measurements within the event's time range
        mask = (patient_spo2['Time'] >= start_time) & (patient_spo2['Time'] <= end_time)
        # Update the labels in the main dataset
        dataset.loc[patient_spo2[mask].index, 'Label'] = event_type
        dataset.loc[patient_spo2[mask].index, 'Type'] = apnea_type


print(dataset.loc[dataset['Label'] == 1])
train_df = dataset[dataset["Patient"].isin(train_patients)].reset_index(drop=True)
test_df = dataset[dataset["Patient"].isin(test_patients)].reset_index(drop=True)

train_df = one_hot_encode_labels(train_df)  # Apply one-hot encoding
test_df = one_hot_encode_labels(test_df)  # Apply one-hot encoding


print(f"Training set: {train_df.shape[0]} rows, {train_df['Patient'].nunique()} patients")
print(f"Test set: {test_df.shape[0]} rows, {test_df['Patient'].nunique()} patients")

train_df.to_feather('datasets/trainset-labeled.feather')
test_df.to_feather('datasets/testset-labeled.feather')

# Generate segments for training and testing separately
#train_segments = generate_segments(train_df, train_patients)
#test_segments = generate_segments(test_df, test_patients) 


#print(f"Training set: {len(train_segments)} segments")
#print(f"Test set: {len(test_segments)} segments")
#print(train_segments['Segment'])




#train_segments.to_feather('datasets/trainset-segments.feather')
#test_segments.to_feather('datasets/testset-segments.feather')