import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt



# Constatns for the segment splits
FS = 128  # Sampling rate (Hz)
WINDOW_SIZE = FS * 60  # 1-minute window (7680 samples)
HALF_WINDOW = WINDOW_SIZE // 2  # 50% overlap
APNEA_RATIO = 0.5  # Desired balance between apnea and non-apnea events
spo2_min_threshold = 68  # Not sure what to use here


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





def get_random_apnea_segment(df):
    apnea_indices = df[df["Label"] == 1].index.to_list()  # Get all apnea row indices


    random_idx = np.random.choice(apnea_indices)  # Pick a random apnea index
    start_idx = random_idx - (WINDOW_SIZE // 2)   # Get start index
    end_idx = random_idx + (WINDOW_SIZE // 2) - 1  # Get end index

    # Ensure we stay within dataset bounds
    if start_idx >= 0 and end_idx < len(df):
        segment = df.iloc[start_idx:end_idx + 1]["SpO2"].values  # Extract SpO2 values
        return segment
    else:
        return -1



# Function to generate sliding window segments
def generate_segments(df, patients):
    segments = []  # Store final segments
    apnea_segments = []  # Store apnea-heavy segments separately

    for patient in patients:
        patient_df = df[df["Patient"] == patient].reset_index(drop=True)

        # Generate sliding window segments (50% overlap)
        for start in range(0, len(patient_df) - WINDOW_SIZE, HALF_WINDOW):
            segment_df = patient_df.iloc[start:start + WINDOW_SIZE]  # Extract segment
            spo2_values = segment_df["SpO2"].values  # Convert to NumPy array
            label = 1 if segment_df["Label"].sum() > 0 else 0  # Label = 1 if apnea present
            segments.append((spo2_values, label))

            # Store apnea segments separately for augmentation
            if label == 1:
                apnea_segments.append(spo2_values)

    print(len(segments), len(apnea_segments))

    # Randomly sample apnea segments to balance the dataset
    count = int((len(segments) - len(apnea_segments)) * APNEA_RATIO)
    for i in range(len(apnea_segments),count):
        random_segment = get_random_apnea_segment(df)
        print(len(segments))
        if isinstance(random_segment,int):
            i -= 1
        else:
            segments.append((random_segment,1))
    #    num_apnea_samples = min(random_apnea_samples, len(apnea_segments))
    #    extra_apnea_segments = np.random.choice(apnea_segments, num_apnea_samples, replace=True)
   #     
        # Append extra apnea cases to dataset
    #    for seg in extra_apnea_segments:
    #        segments.append((seg, 1))
    print(len(segments), len(apnea_segments))
    return pd.DataFrame(segments, columns=["Segment", "Label"])


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
        event_type = 1
        
        # Find SpO2 measurements within the event's time range
        mask = (patient_spo2['Time'] >= start_time) & (patient_spo2['Time'] <= end_time)
        # Update the labels in the main dataset
        dataset.loc[patient_spo2[mask].index, 'Label'] = event_type


train_df = dataset[dataset["Patient"].isin(train_patients)].reset_index(drop=True)
test_df = dataset[dataset["Patient"].isin(test_patients)].reset_index(drop=True)

print(f"Training set: {train_df.shape[0]} rows, {train_df['Patient'].nunique()} patients")
print(f"Test set: {test_df.shape[0]} rows, {test_df['Patient'].nunique()} patients")

# Generate segments for training and testing separately
train_segments = generate_segments(train_df, train_patients)
test_segments = generate_segments(test_df, test_patients)  # No balancing in test set


print(f"Training set: {len(train_segments)} segments")
print(f"Test set: {len(test_segments)} segments")
print(train_segments['Segment'])


train_df.to_feather('datasets/trainset-labeled.feather')
train_df.to_feather('datasets/testset-labeled.feather')