import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from functools import partial
import multiprocessing as mp

# Constants for the segment splits
FS = 128  # Original sampling rate (Hz)
TARGET_FREQUENCY = 8 # Desired sampling frequency (Hz)
LENGTH = 30
#WINDOW_SIZE = int(FS * LENGTH / (FS // TARGET_FREQUENCY))  # Adjusted window size
WINDOW_OVERLAP = 0.5
HALF_WINDOW = int(LENGTH * FS * WINDOW_OVERLAP)  # 50% overlap
APNEA_RATIO = 1.0  # Desired balance between apnea and non-apnea events
SAMPLE_INTERVAL = FS // TARGET_FREQUENCY  # How many original samples to skip

spo2_min_threshold = 80  # Not sure what to use here
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
    
    # Recalculate window size based on target frequency

    # Get all apnea row indices
    apnea_indices = df[df["Label"] == 1].index.to_list()

    # Pick a random apnea index
    random_idx = np.random.choice(apnea_indices)

    # Calculate start and end indices
    start_idx = random_idx - LENGTH*FS//2
    end_idx = random_idx + LENGTH*FS//2 - 1
    segment_indices = range(int(start_idx), int(end_idx), SAMPLE_INTERVAL)
    # Ensure we stay within dataset bounds
    if start_idx >= 0 and end_idx < len(df):
        # Extract full segment
        sampled_segment = df.iloc[segment_indices]["SpO2"].values
        
        
        return sampled_segment, random_idx
    else:
        return sampled_segment, -1

def extract_segment(df, batch_size, counter, lock):
    """Extract multiple apnea segments in a batch with progress tracking."""
    results = []
    process_name = mp.current_process().name
    process_id = int(process_name.split("-")[-1])
    seed = process_id * 1000
    np.random.seed(seed)
    for _ in range(batch_size):
        segment, random_idx = get_random_apnea_segment(df)
        if random_idx != -1:  # Discard invalid segments
            results.append((segment, random_idx))

        # Update progress using shared counter
        #with lock:
        #    counter.value += 1
        #if counter.value % 100 == 0:  # Print progress every 100 segments
        #    print(f"Progress: {counter.value} segments processed")

    print(f"{process_name} process processed  {batch_size} segments", flush=True)
    return results

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
    #sample_interval = FS // target_frequency
    #window_size = int(FS * 60 / sample_interval)
    #half_window = window_size // 2

    for patient in patients:
        patient_df = df[df["Patient"] == patient].reset_index(drop=True)

        # Generate sliding window segments (50% overlap)
        for start in range(0, len(patient_df) - (LENGTH*FS), HALF_WINDOW):
            # Sample every nth point based on target frequency
            segment_indices = range(start, start + (LENGTH*FS), SAMPLE_INTERVAL)
            #print(segment_indices)
            segment_df = patient_df.iloc[segment_indices]

            # Extract sampled SpO2 values
            spo2_values = segment_df["SpO2"].values

            # Labels are now precomputed from the dataset
            binary_label = 1 if segment_df["Label"].sum() > 10*TARGET_FREQUENCY else 0
            apnea_label = 1 if segment_df["Apnea"].sum() > 10*TARGET_FREQUENCY else 0  # If any Apnea in segment, set to 1
            hypopnea_label = 1 if segment_df["Hypopnea"].sum() > 10*TARGET_FREQUENCY else 0   # If any Hypopnea, set to 1
            if apnea_label == 0 and hypopnea_label == 0:
                other_label = segment_df["Other"].max()  # If none, set Other = 1
            else:
                other_label = 0

            segments.append((spo2_values, binary_label, apnea_label, hypopnea_label, other_label))

            # Store apnea segments separately for augmentation
            if binary_label == 1:
                apnea_segments.append(spo2_values)

    count = int((len(segments) - len(apnea_segments)) * APNEA_RATIO)
    print(f"Total segments: {len(segments)}, Apnea segments: {len(apnea_segments)}")

    num_processes = 6  # Number of processes in the pool
    batch_size = 128  # Number of segments to process per batch
    num_batches = (count + batch_size - 1) // batch_size  # Calculate number of batches

    print(f"Starting parallel extraction using {num_processes} processes with batch size {batch_size}")

    # Use a global manager to create a shared counter and lock
    with mp.Manager() as manager:
        counter = manager.Value("i", 0)  # Shared counter
        lock = manager.Lock()  # Lock for thread safety

        # Create a process pool
        with mp.Pool(processes=num_processes) as pool:
            # Submit batches of work
            results = [
                pool.apply_async(extract_segment, args=(df, batch_size, counter, lock))
                for _ in range(num_batches)
            ]

            # Collect results
            for r in results:
                batch_segments = r.get()  # Get the results from this batch
                for segment, idx in batch_segments:
                    if segment is not None:  # Only add valid segments
                        segments.append((segment, 1 , df.iloc[idx]['Apnea'], 
                             df.iloc[idx]['Hypopnea'], 
                             df.iloc[idx]['Other']))

    print(f"Final segments: {len(segments)}")

    return pd.DataFrame(segments, columns=["Segment", "Label", "Apnea", "Hypopnea", "Other"])



train = 'datasets/trainset-labeled.feather'
test = 'datasets/testset-labeled.feather'
pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

train = pd.read_feather(train)  
test = pd.read_feather(test)



# Generate segments for training and testing separately
train_segments = generate_segments(train, train_patients)
test_segments = generate_segments(test, test_patients) 


print(f"Training set: {len(train_segments)} segments")
print(f"Test set: {len(test_segments)} segments")



train_segments.to_feather('datasets/trainset-segments.feather')
test_segments.to_feather('datasets/testset-segments.feather')