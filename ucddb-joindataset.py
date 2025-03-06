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

#patient split for train/test
train_patients = [
    "ucddb002", "ucddb003", "ucddb006", "ucddb007", "ucddb009",
    "ucddb010", "ucddb011", "ucddb012", "ucddb013", "ucddb017", "ucddb018", "ucddb019", "ucddb020", "ucddb022",
    "ucddb023", "ucddb025", "ucddb027", "ucddb028"
]

test_patients = [
    "ucddb005", "ucddb014", "ucddb015", "ucddb021", "ucddb024",
    "ucddb008", "ucddb026"
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


def get_random_apnea_segment(df):
    """
    Get a random apnea segment with sampling at specified frequency.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    target_frequency (int): Desired sampling frequency in Hz. Default is 1 Hz.
    original_fs (int): Original sampling frequency. Default is 128 Hz.
    
    Returns:
    Tuple: (sampled segment, random index) or -1 if segment cannot be extracted
    """
    # Calculate sampling interval
    sample_interval = FS // TARGET_FREQUENCY
    
    # Recalculate window size based on target frequency

    # Get all apnea row indices
    apnea_indices = df[df["Label"] == 1].index.to_list()

    # Pick a random apnea index
    random_idx = np.random.choice(apnea_indices)

    # Calculate start and end indices
    start_idx = random_idx - HALF_WINDOW
    end_idx = random_idx + HALF_WINDOW - 1
    segment_indices = range(start_idx, end_idx, sample_interval)
    # Ensure we stay within dataset bounds
    if start_idx >= 0 and end_idx < len(df):
        # Extract full segment
        sampled_segment = df.iloc[segment_indices]["SpO2"].values
        
        
        return sampled_segment, random_idx
    else:
        return sampled_segment, -1

def generate_segments(df, patients, target_frequency=1):
    """
    Extracts sliding window segments with specified sampling frequency and balances apnea cases.
    
    Parameters:
        df (pd.DataFrame): Preprocessed DataFrame with SpO₂ values and labels.
        patients (list): List of patient IDs to process.
        target_frequency (int): Desired sampling frequency in Hz. Default is 1 Hz.
    
    Returns:
        pd.DataFrame: DataFrame containing extracted segments.
    """
    segments = []  # Store final segments
    apnea_segments = []  # Store apnea-heavy segments separately

    # Recalculate window size and sampling interval based on target frequency
    sample_interval = FS // target_frequency
    window_size = int(FS * 60 / sample_interval)
    half_window = window_size // 2

    for patient in patients:
        patient_df = df[df["Patient"] == patient].reset_index(drop=True)

        # Generate sliding window segments (50% overlap)
        for start in range(0, len(patient_df) - window_size, half_window):
            # Sample every nth point based on target frequency
            segment_indices = range(start, start + window_size, sample_interval)
            segment_df = patient_df.iloc[segment_indices]

            # Extract sampled SpO2 values
            spo2_values = segment_df["SpO2"].values

            # Labels are now precomputed from the dataset
            binary_label = 1 if segment_df["Label"].sum() > 0 else 0
            apnea_label = segment_df["Apnea"].max()  # If any Apnea in segment, set to 1
            hypopnea_label = segment_df["Hypopnea"].max()  # If any Hypopnea, set to 1
            if apnea_label == 0 and hypopnea_label == 0:
                other_label = segment_df["Other"].max()  # If none, set Other = 1
            else:
                other_label = 0

            segments.append((spo2_values, binary_label, apnea_label, hypopnea_label, other_label))

            # Store apnea segments separately for augmentation
            if binary_label == 1:
                apnea_segments.append(spo2_values)

    print(f"Total segments: {len(segments)}, Apnea segments: {len(apnea_segments)}")

    # Randomly sample apnea segments to balance the dataset
    count = int((len(segments) - len(apnea_segments)) * APNEA_RATIO)
    for i in range(count):
        random_segment, idx = get_random_apnea_segment(df)
        if idx < 0:  # Check if function failed to find a segment
            i -= 1
        else:
            # Resample the random segment to match target frequency
            random_segment_sampled = random_segment[::sample_interval]
            segments.append((random_segment_sampled, 1, 
                             df.iloc[idx]['Apnea'], 
                             df.iloc[idx]['Hypopnea'], 
                             df.iloc[idx]['Other']))

    print(f"Final segments: {len(segments)}")
    return pd.DataFrame(segments, columns=["Segment", "Label", "Apnea", "Hypopnea", "Other"])

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