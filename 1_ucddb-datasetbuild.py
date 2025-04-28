import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import glob



def read_rec(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.readlines()
        return data
    except Exception as e:
        print(f"Error reading REC file: {e}")
        return None

def convert_measure_time(info):
    meas_date = info['meas_date']
    if isinstance(meas_date, tuple):  # Handle MNE's tuple format
        meas_date = meas_date[0]
    if not isinstance(meas_date, datetime.datetime):
        meas_date = datetime.datetime.fromtimestamp(meas_date, tz=datetime.timezone.utc)

    return meas_date
def load_respiratory_events(file_path, meas_date, patient):
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
                try:
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
                except:
                    time_str = parts[0]  # Time in HH:MM:SS format
                    event_type = parts[1]
                    duration = int(parts[4])
                    low = float(parts[5]) if isinstance(parts[5],float) else None
                    percent_drop = float(parts[6]) if isinstance(parts[6],float)  else None
                    # Convert HH:MM:SS to a datetime object (using the same date as meas_date)
                    hh, mm, ss = map(int, time_str.split(':'))
                    event_datetime = meas_date.replace(hour=hh, minute=mm, second=ss)
                    if meas_date > event_datetime:
                        event_datetime += datetime.timedelta(days=1)
                    time_diff = (event_datetime - meas_date).total_seconds()
                data.append([patient, time_diff, event_type, duration, low, percent_drop])
    
    # Create DataFrame
    columns = ['Patient','Time', 'Type', 'Duration', 'Low', 'PercentDrop']
    df = pd.DataFrame(data, columns=columns)
    
    return df
   
patients = []
resp_events = pd.DataFrame()
patients_df = None

for file in glob.glob("datasets/files/*.edf"):
    p = file.split("/")
    p = p[-1].split(".")
    p = p[0]
    patients.append((p,file))





for p in patients:
    p_name = p[0]
    p_file = p[1]
    data = mne.io.read_raw_edf(p_file)
    raw_data = data.get_data()

    info = data.info
    channels = data.ch_names

    meas_date = convert_measure_time(info)
    p_events = p_file.split(".")[0]
    print(p_name)
    df = load_respiratory_events(p_events + '_respevt.txt',meas_date, p_name)
    resp_events = pd.concat([resp_events,df])

    sfreq = info['sfreq']  # Sampling frequency
    time_vector_seconds = np.arange(raw_data.shape[1]) / sfreq
    

    if isinstance(patients_df, pd.DataFrame):
        df_edf = pd.DataFrame(raw_data.T, columns=channels)
        
        df_edf['Time'] = time_vector_seconds
        df_edf['Patient'] = p_name
        patients_df = pd.concat([patients_df,df_edf],ignore_index=True)
    else:
        patients_df = pd.DataFrame(raw_data.T, columns=channels)
        patients_df['Time'] = time_vector_seconds
        patients_df['Patient'] = p_name

        
   

#print(resp_events[resp_events['Time'] >= 1060.0 and resp_events['Time'] < 1080])
#print(resp_events)

smaller_df = patients_df[['Patient', 'Time', 'SpO2']]
print(smaller_df.tail())

smaller_df.to_feather('datasets/dataset_UCDDB.feather')
resp_events.to_feather('datasets/dataset_UCDDB_events.feather')