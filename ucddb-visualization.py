
import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

def load_respiratory_events(file_path, meas_date):
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
            
            try:
                # Try parsing with the first format
                time_str = parts[0]  # Time in HH:MM:SS format
                event_type = parts[1]
                
                # Handle duration and low/percent drop parsing more robustly
                if len(parts) >= 5:
                    duration = int(parts[2])
                    low = float(parts[3]) if parts[3] != '-' and parts[3] != '+' else None
                    percent_drop = float(parts[4]) if parts[4] != '-' and parts[4] != '+'else None
                else:
                    # Alternative parsing if columns are different
                    duration = int(parts[4])
                    low = float(parts[5]) if parts[5] != '-' and parts[5] != '+' else None
                    percent_drop = float(parts[6]) if parts[6] != '-' and parts[6] != '+' else None
                
                # Convert time to datetime
                hh, mm, ss = map(int, time_str.split(':'))
                event_datetime = meas_date.replace(hour=hh, minute=mm, second=ss)
                
                # Handle date rollover
                if event_datetime < meas_date:
                    event_datetime += datetime.timedelta(days=1)
                
                data.append([event_datetime, event_type, duration, low, percent_drop])
            
            except Exception as e:
                print(f"Error parsing line: {line.strip()}. Error: {e}")
    
    # Create DataFrame
    columns = ['Time', 'Type', 'Duration', 'Low', 'PercentDrop']
    df = pd.DataFrame(data, columns=columns)
    
    return df

# Rest of the code remains the same as in your original script
patient_path = "datasets/files/ucddb027"
file = patient_path + ".edf"
data = mne.io.read_raw_edf(file)
raw_data = data.get_data()
info = data.info
channels = data.ch_names
print(raw_data.shape[1]/128)
# Extract meas_date
meas_date = info['meas_date']
if isinstance(meas_date, tuple):  # Handle MNE's tuple format
    meas_date = meas_date[0]

# Ensure meas_date is a datetime object
if not isinstance(meas_date, datetime.datetime):
    meas_date = datetime.datetime.fromtimestamp(meas_date, tz=datetime.timezone.utc)

# Load respiratory events
resp_events = load_respiratory_events(patient_path + '_respevt.txt', meas_date)
print(resp_events)

# Create DataFrame for EDF data with datetime index
sfreq = 128  # Sampling frequency
num_samples = raw_data.shape[1]  # Total number of samples
time_vector = pd.date_range(start=meas_date, periods=num_samples, freq=pd.Timedelta(seconds=1/sfreq))

df_edf = pd.DataFrame(raw_data.T, columns=channels)
df_edf['Time'] = time_vector

# Plotting
plt.figure(figsize=(15, 6))

# Plot SpO2 data
plt.plot(df_edf["Time"], df_edf["SpO2"], label="SpO₂", color="blue")

# Highlight abnormal sections
for _, row in resp_events.iterrows():
    plt.axvspan(row["Time"], 
                row["Time"] + pd.Timedelta(seconds=row["Duration"]),
                color="red", alpha=0.3, 
                label="Abnormal Event" if _ == 0 else "")

plt.xlabel("Time")
plt.ylabel("SpO₂ (%)")
plt.title("SpO₂ Over Time with Respiratory Events")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()



# Plot EDF data
# Create a figure with two subplots (vertical stack)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# --- Subplot 1: EDF Data with Respiratory Events ---
ax1.plot(df_edf['Time'], df_edf['SpO2'], color='blue', label='EDF SpO2', linewidth=1)
# Overlay EDF respiratory events (from resp_events)
for _, event in resp_events.iterrows():
    ax1.axvspan(event['Time'], event['Time'] - event['Duration'], 
                color='red', alpha=0.3, label='EDF Events' if _ == 0 else "")
ax1.set_title("EDF Data with Respiratory Annotations")
ax1.set_ylabel("SpO2 (%)")
ax1.legend(loc='upper right')

# --- Subplot 2: Labeled Dataset with Abnormal Breathing ---
ax2.plot(dataset['Time'], dataset['SpO2'], color='green', label='Dataset SpO2', linewidth=1)
# Overlay labeled events (where Label == 1)
label_changes = np.diff(dataset['Label'], prepend=0)
event_starts = dataset['Time'][label_changes == 1].values
event_ends = dataset['Time'][np.roll(label_changes, -1) == -1].values
for start, end in zip(event_starts, event_ends):
    ax2.axvspan(start, end, color='purple', alpha=0.3, 
                label='Abnormal Breathing' if start == event_starts[0] else "")
ax2.set_title("Dataset with Labeled Abnormal Breathing")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("SpO2 (%)")
ax2.legend(loc='upper right')


# Sync x-axis and adjust layout
plt.tight_layout()
plt.show()