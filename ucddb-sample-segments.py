import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import os
from pathlib import Path
import itertools
# Original constants
FS = 128  # Original sampling rate (Hz)
APNEA_RATIO = 1.0  # Desired balance between apnea and non-apnea events

# Define different configurations for ablation study
TARGET_FREQUENCIES = [1, 4, 8, 16]  # Hz
SEGMENT_LENGTHS = [ 11, 15, 20, 25, 30]  # seconds
WINDOW_OVERLAPS = [0.25, 0.5, 0.75]  # 25%, 50%, 75% overlap
EVENT_THRESHOLD = 10  # seconds - minimum duration for relevant events

apnea_types = ['APNEA-O', 'APNEA-C', 'APNEA-M']
hypopnea_types = ['HYP-O', 'HYP-C', 'HYP-M']
other_types = ['PB', 'POSSIBLE']



# Patient split for train/test (same as original)
train_patients = [
    "ucddb002", "ucddb003", "ucddb006", "ucddb007", "ucddb009",
    "ucddb010", "ucddb012", "ucddb019", "ucddb020", "ucddb022",
    "ucddb023", "ucddb025", "ucddb027", "ucddb028"
]

test_patients = [
    "ucddb005", "ucddb014", "ucddb015", "ucddb021", "ucddb024",
    "ucddb026"
]

def has_consecutive_events(event_series, threshold):
    """
    Checks if there is a sequence of at least 'threshold' consecutive ones in a binary array.
    
    Parameters:
    -----------
    event_series : numpy array
        Binary array where 1 represents an apnea/hypopnea event, and 0 represents normal.
    threshold : int
        Minimum number of consecutive 1s required for a valid event.
    
    Returns:
    --------
    bool : True if threshold is met, False otherwise.
    """
    # Group consecutive 1s and count their lengths
    for key, group in itertools.groupby(event_series):
        if key == 1 and sum(1 for _ in group) >= threshold:
            return True  # Found a valid event
    return False  # No valid event found

def generate_segments(df, patients, length, target_frequency, window_overlap, balance_method='duplicate'):

    segments = []  # Store all segments
    apnea_segments = []  # Store apnea segments separately

    # Calculate derived parameters
    sample_interval = FS // target_frequency
    window_size = length * FS
    step_size = int(window_size * (1 - window_overlap))  # Convert overlap to step size

    # Event detection threshold in samples
    event_threshold_samples = EVENT_THRESHOLD * target_frequency

    # Process each patient
    for patient in patients:
        patient_df = df[df["Patient"] == patient].reset_index(drop=True)

        # Generate sliding window segments with overlap
        for start in tqdm(range(0, len(patient_df) - window_size, step_size), desc="Processing"):
            # Sample every nth point based on target frequency
            segment_indices = range(start, start + window_size, sample_interval)
            segment_df = patient_df.iloc[segment_indices]
            # Extract sampled SpO2 values
            spo2_values = segment_df["SpO2"].values

            # Convert label sequence to binary (0s and 1s)
            label_sequence = segment_df["Label"].values.astype(int)

            # Check for at least 'event_threshold_samples' consecutive 1s
            binary_label = 1 if has_consecutive_events(label_sequence, event_threshold_samples) else 0

            # Repeat for apnea and hypopnea
            apnea_sequence = segment_df["Apnea"].values.astype(int)
            hypopnea_sequence = segment_df["Hypopnea"].values.astype(int)
            
            apnea_label = 1 if has_consecutive_events(apnea_sequence, event_threshold_samples) else 0
            hypopnea_label = 1 if has_consecutive_events(hypopnea_sequence, event_threshold_samples) else 0
            
            if apnea_label == 0 and hypopnea_label == 0:
                other_label = segment_df["Other"].max()
            else:
                other_label = 0

            segment_data = (spo2_values, binary_label, apnea_label, hypopnea_label, other_label)
            segments.append(segment_data)
                
            # Store positive samples separately for balancing
            if binary_label == 1:
                apnea_segments.append(segment_data)

    # Balance dataset if needed
    balanced_segments = balance_dataset(segments, apnea_segments, balance_method)
    
    # Create DataFrame from segments
    result_df = pd.DataFrame(balanced_segments, columns=["Segment", "Label", "Apnea", "Hypopnea", "Other"])
    
    # Add class weights if using weight method
    if balance_method == 'weight':
        # Calculate class weights inversely proportional to class frequencies
        class_counts = result_df['Label'].value_counts()
        total = len(result_df)
        weights = {}
        for cls, count in class_counts.items():
            weights[cls] = total / (len(class_counts) * count)
        
        # Add weight column
        result_df['Weight'] = result_df['Label'].map(weights)
    
    return result_df

def balance_dataset(segments, apnea_segments, method):

    print(f"Total segments: {len(segments)}, Apnea segments: {len(apnea_segments)}")
    
    # Return original segments if no balancing requested
    if method == 'none' or not apnea_segments:
        print("No balancing performed")
        return segments
    
    # Calculate target number of apnea segments based on APNEA_RATIO
    non_apnea_count = len(segments) - len(apnea_segments)
    target_apnea_count = int(non_apnea_count * APNEA_RATIO)
    
    if method == 'duplicate':
        # Simple duplication of existing apnea segments
        if target_apnea_count <= len(apnea_segments):
            # We already have enough apnea segments
            print("No duplication needed - already balanced")
            return segments
        
        # How many times to duplicate each apnea segment on average
        duplication_factor = target_apnea_count / len(apnea_segments)
        full_copies = int(duplication_factor)
        remainder = duplication_factor - full_copies
        
        # Add full copies of all apnea segments
        additional_segments = []
        for _ in range(full_copies):
            additional_segments.extend(apnea_segments)
        
        # Add partial copies for remainder
        remainder_count = int(remainder * len(apnea_segments))
        if remainder_count > 0:
            # Randomly select segments to duplicate
            np.random.seed(42)  # For reproducibility
            remainder_indices = np.random.choice(
                range(len(apnea_segments)), 
                size=remainder_count, 
                replace=False
            )
            for idx in remainder_indices:
                additional_segments.append(apnea_segments[idx])
        
        print(f"Added {len(additional_segments)} duplicate apnea segments")
        return segments + additional_segments
    
    elif method == 'weight':
        # No need to add segments when using weights
        print("Using class weights for balancing")
        return segments
    
    else:
        print(f"Unknown balancing method: {method}, returning original segments")
        return segments

def generate_ablation_datasets():
    """Generate multiple datasets for ablation study."""
    # Load source datasets
    train_path = 'datasets/trainset-labeled.feather'
    test_path = 'datasets/testset-labeled.feather'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Source datasets not found at {train_path} or {test_path}")
        return
    
    train = pd.read_feather(train_path)
    test = pd.read_feather(test_path)
    
    # Create output directory for ablation datasets
    output_dir = Path('datasets/ablation_study')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Track dataset statistics for summary
    dataset_stats = []
    
    # Generate datasets for each configuration
    for freq in TARGET_FREQUENCIES:
        for length in SEGMENT_LENGTHS:
            for overlap in WINDOW_OVERLAPS:
                print(f"\n{'='*50}")
                print(f"Generating datasets with frequency={freq}Hz, length={length}s, overlap={overlap:.2f}")
                print(f"{'='*50}")
                
                # Define output filenames
                train_output = output_dir / f"train_freq{freq}_len{length}_overlap{int(overlap*100)}.feather"
                test_output = output_dir / f"test_freq{freq}_len{length}_overlap{int(overlap*100)}.feather"
                
                # Skip if files already exist
                if train_output.exists() and test_output.exists():
                    print(f"Datasets already exist for freq={freq}Hz, length={length}s, overlap={overlap:.2f}. Skipping...")
                    
                    # Load existing files to get stats
                    train_segments = pd.read_feather(train_output)
                    test_segments = pd.read_feather(test_output)
                    
                    dataset_stats.append({
                        'frequency': freq,
                        'length': length,
                        'overlap': overlap,
                        'train_segments': len(train_segments),
                        'train_apnea': train_segments['Label'].sum(),
                        'train_apnea_pct': train_segments['Label'].mean() * 100,
                        'test_segments': len(test_segments),
                        'test_apnea': test_segments['Label'].sum(),
                        'test_apnea_pct': test_segments['Label'].mean() * 100,
                        'sequence_length': len(train_segments['Segment'].iloc[0])
                    })
                    continue
                
                # Generate segments - use simple duplication for train, no balancing for test
                train_segments = generate_segments(train, train_patients, length, freq, overlap, balance_method='none')
                test_segments = generate_segments(test, test_patients, length, freq, overlap, balance_method='none')
                
                # Save datasets as feather files
                train_segments.to_feather(train_output)
                test_segments.to_feather(test_output)
                
                # Record statistics
                dataset_stats.append({
                    'frequency': freq,
                    'length': length,
                    'overlap': overlap,
                    'train_segments': len(train_segments),
                    'train_apnea': train_segments['Label'].sum(),
                    'train_apnea_pct': train_segments['Label'].mean() * 100,
                    'test_segments': len(test_segments),
                    'test_apnea': test_segments['Label'].sum(),
                    'test_apnea_pct': test_segments['Label'].mean() * 100,
                    'sequence_length': len(train_segments['Segment'].iloc[0]),
                    'balance_method': 'duplicate'
                })
                
                print(f"Saved datasets to {train_output} and {test_output}")
    
    # Create summary dataframe and save
    summary_df = pd.DataFrame(dataset_stats)
    summary_path = output_dir / "ablation_summary.feather"
    summary_df.to_feather(summary_path)  # Save as feather
    
    # Also save as CSV for easy viewing
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)
    
    print(f"\nSummary saved to {summary_path}")
    print(summary_df)
    
    return summary_df

if __name__ == "__main__":
    # Generate all ablation study datasets
    summary = generate_ablation_datasets()