
import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime




def read_rec(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.readlines()
        return data
    except Exception as e:
        print(f"Error reading REC file: {e}")
        return None


def load_respiratory_events(file_path, date):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the start of the data
    start_idx = next(i for i, line in enumerate(lines) if "Time" in line)
    
    # Read the data into a DataFrame
    data = []
    for line in lines[start_idx+1:]:
        if line.strip():
            parts = line.split()
            if len(parts) > 1:
                time_str = parts[0]  # Time in HH:MM:SS format
                event_type = parts[1]
                duration = int(parts[2])
                low = float(parts[3]) if isinstance(parts[3],float) else None
                percent_drop = float(parts[4]) if isinstance(parts[4],float)  else None
                # Convert HH:MM:SS to a datetime object (using the same date as meas_date)
                hh, mm, ss = map(int, time_str.split(':'))
                event_datetime = meas_date.replace(hour=hh, minute=mm, second=ss)
                if meas_date > event_datetime:
                    event_datetime += datetime.timedelta(days=1)
                # Calculate the time difference in seconds from meas_date
                time_diff = (event_datetime - meas_date).total_seconds()
                
                data.append([time_diff, event_type, duration, low, percent_drop])
    
    # Create DataFrame
    columns = ['Time', 'Type', 'Duration', 'Low', 'PercentDrop']
    df = pd.DataFrame(data, columns=columns)
    
    return df
                
patient_path = "datasets/files/ucddb002"

file = patient_path + ".edf"
data = mne.io.read_raw_edf(file)
raw_data = data.get_data()
# you can get the metadata included in the file and a list of all channels:
info = data.info
channels = data.ch_names


print(info)


# Extract meas_date
meas_date = info['meas_date']
if isinstance(meas_date, tuple):  # Handle MNE's tuple format
    meas_date = meas_date[0]

# Ensure meas_date is a datetime object
if not isinstance(meas_date, datetime.datetime):
    meas_date = datetime.datetime.fromtimestamp(meas_date, tz=datetime.timezone.utc)


resp_events = load_respiratory_events(patient_path + '_respevt.txt',meas_date)
print(resp_events.head())




#print(channels)


# Create a DataFrame
df = pd.DataFrame(raw_data.T, columns=channels)
# Create a time vector in seconds
sfreq = info['sfreq']  # Sampling frequency
time_vector_seconds = np.arange(raw_data.shape[1]) / sfreq

# Create DataFrame for EDF data
df_edf = pd.DataFrame(raw_data.T, columns=channels)
df_edf['Time'] = time_vector_seconds  # Use seconds for time

# Print the first few rows of the DataFrame
#print(df_edf.head())


# Plot EDF data
plt.figure(figsize=(14, 10))
for i, channel in enumerate(channels[5:10]): 
    plt.subplot(5, 1, i+1)
    plt.plot(df_edf['Time'], df_edf[channel], label=channel)
    
    # Overlay respiratory events
    for _, event in resp_events.iterrows():
        event_start = event['Time']
        event_end = event_start + event['Duration']
        plt.axvspan(event_start, event_end, color='red', alpha=0.3)
    
    plt.title(channel)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()

plt.tight_layout()
plt.show()
