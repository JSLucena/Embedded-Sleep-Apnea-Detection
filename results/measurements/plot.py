import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define inference windows for each configuration
inference_windows_map = {
    # Pico2 configurations
    'p2c': [(0.336, 0.341), (1.494,1.499), (2.77, 2.775), (4.067, 4.073)],  # Pico2 cmsis 10infs -> 42ms
    'p2q': [(0.448, 0.544), (1.549,1.645), (3.036,3.132), (4.46, 4.555)],    # Pico2 QAT 10infs -> 977ms
    'p2i': [(0.875, 0.975), (2.695,2.8), (4.532, 4.628)],                    # Pico2 INT8 10infs -> 977ms
    'p2': [(0.237, 0.267), (1.612,1.639), (2.593,2.623), (3.687, 3.72)],    # Pico2 baseline 10infs -> 225ms
    
    # Pico configurations
    'p1c': [(0.878, 0.898), (1.557,1.576), (2.605,2.623), (3.453, 3.472), (4.224, 4.242)],  # Pico cmsis 10infs -> 182ms
    'p1q': [(0.753, 0.946), (2.156,2.35), (3.842,4.034)],                                   # Pico QAT 10infs -> 1970ms
    'p1i': [(0.415, 0.609), (1.927,2.12), (3.545,3.739)],                                   # pico int8 10infs -> 1969ms
    'p1': [(0.44, 0.7), (2.27,2.53), (4.117,4.38)]                                        # Pico baselne 10infs -> 2651ms
}

def read_n6705c_datalog(filename, sample_interval=0.0001):  # 0.1ms
    data = pd.read_csv(filename)

    data['Current_mA'] = data['Curr avg 1'] * 1000  # A to mA
    data['Power_mW'] = data['Volt avg 1'] * data['Current_mA']  # mW
    data['Time'] = data['Sample'] * sample_interval  # Time in seconds
    return data

def moving_average_robust(x, w):
    """
    Robust moving average with edge padding to avoid artifacts
    """
    if w <= 1:
        return x
    
    # Pad the signal at edges to reduce edge effects
    half_window = w // 2
    padded = np.pad(x, (half_window, half_window), mode='edge')
    
    # Apply convolution and trim to original length
    smoothed = np.convolve(padded, np.ones(w), 'valid') / w
    
    # Ensure output length equals input length
    if len(smoothed) != len(x):
        # Adjust for odd/even window sizes
        start_idx = (len(smoothed) - len(x)) // 2
        smoothed = smoothed[start_idx:start_idx + len(x)]
    
    return smoothed

def validate_sample_rate(data, expected_interval):
    """
    Validate the actual sample rate matches expected
    """
    if len(data) < 2:
        return expected_interval, True
    
    actual_intervals = np.diff(data['Time'])
    actual_interval = np.mean(actual_intervals)
    interval_std = np.std(actual_intervals)
    
    # Check if interval is consistent
    is_consistent = interval_std < (actual_interval * 0.01)  # 1% tolerance
    
    # Check if it matches expected
    matches_expected = abs(actual_interval - expected_interval) < (expected_interval * 0.05)  # 5% tolerance
    
    return actual_interval, matches_expected and is_consistent

def validate_inference_windows(data, inference_windows, power_column='Power_mW'):
    """
    Validate that inference windows align with actual power spikes
    """
    if not inference_windows:
        return True, []
    
    warnings = []
    baseline_power = data[power_column].quantile(0.1)  # Use 10th percentile as baseline
    spike_threshold = baseline_power * 1.5  # 50% above baseline considered a spike
    
    for i, (start, end) in enumerate(inference_windows):
        # Check if window is within data range
        if start < data['Time'].min() or end > data['Time'].max():
            warnings.append(f"Window {i+1} ({start:.3f}-{end:.3f}s) is outside data range")
            continue
        
        # Get power data in this window
        mask = (data['Time'] >= start) & (data['Time'] <= end)
        window_power = data.loc[mask, power_column]
        
        if len(window_power) == 0:
            warnings.append(f"Window {i+1} contains no data points")
            continue
        
        # Check if there's actually a power spike in this window
        max_power_in_window = window_power.max()
        if max_power_in_window < spike_threshold:
            warnings.append(f"Window {i+1}: No significant power spike detected (max={max_power_in_window:.1f}mW, threshold={spike_threshold:.1f}mW)")
    
    is_valid = len(warnings) == 0
    return is_valid, warnings

def calculate_robust_baseline(data, inference_windows, min_idle_duration=0.1):
    """
    Calculate baseline power with validation of idle periods
    """
    if not inference_windows:
        return data['Power_mW'].mean(), data['Current_mA'].mean(), True
    
    # Create mask for non-inference periods
    outside_inference_mask = np.ones(len(data), dtype=bool)
    for (start, end) in inference_windows:
        outside_inference_mask &= ~((data['Time'] >= start) & (data['Time'] <= end))
    
    # Check if we have sufficient idle time
    idle_data = data.loc[outside_inference_mask]
    if len(idle_data) == 0:
        return np.nan, np.nan, False
    
    # Calculate total idle duration
    idle_time_total = len(idle_data) * (data['Time'].iloc[1] - data['Time'].iloc[0])
    
    sufficient_idle = idle_time_total >= min_idle_duration
    
    avg_power_idle = idle_data['Power_mW'].mean()
    avg_current_idle = idle_data['Current_mA'].mean()
    
    return avg_power_idle, avg_current_idle, sufficient_idle

def get_inference_windows(filename):
    # Extract the base filename without path and extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    # Remove any numbers from the end (like p2c1, p2c2, etc.)
    clean_name = ''.join([c for c in base_name if not c.isdigit()])
    return inference_windows_map.get(clean_name, inference_windows_map.get(base_name, []))

# Main analysis
def main():
    print("Power Analysis Script - Enhanced Version")
    print("=" * 50)
    
    # Configuration
    sample_interval = 0.0001  # 0.1ms
    input_file = 'results/measurements/p2.csv'  # Change this to your input file
    window_size = 10  # 1ms smoothing window
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Analyzing file: {input_file}")
    
    # Read data
    try:
        data = read_n6705c_datalog(input_file, sample_interval=sample_interval)
        print(f"Successfully loaded {len(data)} data points")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Validate sample rate
    actual_interval, rate_valid = validate_sample_rate(data, sample_interval)
    print(f"\nSample Rate Validation:")
    print(f"Expected interval: {sample_interval:.6f} s")
    print(f"Actual interval: {actual_interval:.6f} s")
    print(f"Rate validation: {'✓ PASS' if rate_valid else '✗ FAIL'}")
    
    if not rate_valid:
        print(f"Warning: Using actual interval ({actual_interval:.6f} s) for calculations")
        sample_interval = actual_interval
    
    # Get inference windows
    inference_windows = get_inference_windows(input_file)
    if not inference_windows:
        print(f"Warning: No inference windows defined for file {input_file}")
    else:
        print(f"Found {len(inference_windows)} inference windows")
    
    # Validate inference windows
    if inference_windows:
        windows_valid, window_warnings = validate_inference_windows(data, inference_windows)
        print(f"\nInference Window Validation:")
        if windows_valid:
            print("✓ All inference windows appear valid")
        else:
            print("✗ Issues found with inference windows:")
            for warning in window_warnings:
                print(f"  - {warning}")
    
    # Apply robust smoothing
    print(f"\nApplying smoothing (window size: {window_size} samples = {window_size * sample_interval * 1000:.1f} ms)")
    data['Current_mA_smoothed'] = moving_average_robust(data['Current_mA'], window_size)
    data['Power_mW_smoothed'] = moving_average_robust(data['Power_mW'], window_size)
    
    # Calculate robust baseline
    avg_power_idle, avg_current_idle, sufficient_idle = calculate_robust_baseline(
        data, inference_windows, min_idle_duration=0.1
    )
    
    # Plotting
    plt.figure(figsize=(14, 10))
    
    # Current plot
    plt.subplot(2, 1, 1)
    #plt.plot(data['Time'], data['Current_mA'], 'b-', alpha=0.3, label='Raw Current')
    plt.plot(data['Time'], data['Current_mA_smoothed'], 'r-', linewidth=2, label='Current')
    if not np.isnan(avg_current_idle):
        plt.axhline(y=avg_current_idle, color='orange', linestyle='--', alpha=0.7, label=f'Idle Avg ({avg_current_idle:.1f} mA)')
    plt.title('Current Consumption')
    plt.ylabel('Current (mA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Power plot
    plt.subplot(2, 1, 2)
    #plt.plot(data['Time'], data['Power_mW'], 'b-', alpha=0.3, label='Raw Power')
    plt.plot(data['Time'], data['Power_mW_smoothed'], 'g-', linewidth=2, label='Power')
    if not np.isnan(avg_power_idle):
        plt.axhline(y=avg_power_idle, color='orange', linestyle='--', alpha=0.7, label=f'Idle Avg ({avg_power_idle:.1f} mW)')
    
    # Highlight inference windows
    for i, (start, end) in enumerate(inference_windows):
        plt.axvspan(start, end, color='red', alpha=0.2, label='Inference Window' if i == 0 else "")
    
    plt.title('Power Consumption')
    plt.ylabel('Power (mW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # CPU-only power plot (if we have baseline)
    """
    if not np.isnan(avg_power_idle):
        plt.subplot(3, 1, 3)
        cpu_only_power = data['Power_mW_smoothed'] - avg_power_idle
        plt.plot(data['Time'], cpu_only_power, 'purple', linewidth=2, label='CPU-only Power')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Highlight inference windows
        for i, (start, end) in enumerate(inference_windows):
            plt.axvspan(start, end, color='red', alpha=0.2, label='Inference Window' if i == 0 else "")
        
        plt.title('CPU-Only Power (Total - Idle)')
        plt.xlabel('Time (s)')
        plt.ylabel('Power (mW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    """
    plt.tight_layout()
    plt.show()
    
    # Statistics with confidence intervals
    print("\n" + "="*50)
    print("MEASUREMENT STATISTICS")
    print("="*50)
    
    print(f"Average Voltage: {data['Volt avg 1'].mean():.6f} ± {data['Volt avg 1'].std():.6f} V")
    print(f"Average Current (raw): {data['Current_mA'].mean():.3f} ± {data['Current_mA'].std():.3f} mA")
    print(f"Average Power (raw): {data['Power_mW'].mean():.3f} ± {data['Power_mW'].std():.3f} mW")
    print(f"Sample Interval: {sample_interval:.6f} seconds")
    print(f"Total Duration: {data['Time'].iloc[-1]:.3f} seconds")
    print(f"Total Samples: {len(data)}")
    
    if not np.isnan(avg_power_idle):
        print(f"\n{'='*20} BASELINE (IDLE) {'='*20}")
        idle_mask = np.ones(len(data), dtype=bool)
        for (start, end) in inference_windows:
            idle_mask &= ~((data['Time'] >= start) & (data['Time'] <= end))
        
        idle_data = data.loc[idle_mask]
        idle_current_std = idle_data['Current_mA'].std()
        idle_power_std = idle_data['Power_mW'].std()
        
        print(f"Average Current (idle): {avg_current_idle:.3f} ± {idle_current_std:.3f} mA")
        print(f"Average Power (idle): {avg_power_idle:.3f} ± {idle_power_std:.3f} mW")
        print(f"Idle data points: {len(idle_data)} ({len(idle_data)/len(data)*100:.1f}% of total)")
        print(f"Sufficient idle time: {'✓' if sufficient_idle else '✗'}")
        
        if inference_windows:
            print(f"\n{'='*15} CPU-ONLY INFERENCE ANALYSIS {'='*15}")
            total_cpu_energy = 0
            total_inference_time = 0
            
            for i, (start, end) in enumerate(inference_windows):
                # Mask for this inference window
                mask = (data['Time'] >= start) & (data['Time'] <= end)
                window_data = data.loc[mask]
                duration = end - start
                
                if len(window_data) == 0:
                    print(f"\nInference {i+1}: No data points found!")
                    continue
                
                # Total energy (including idle power) using trapezoidal integration
                total_energy_mJ = np.trapz(window_data['Power_mW'], dx=sample_interval)
                
                # Subtract idle energy to get CPU-only energy
                idle_energy_mJ = avg_power_idle * duration
                cpu_energy_mJ = total_energy_mJ - idle_energy_mJ
                
                # CPU-only average power
                cpu_avg_power_mW = cpu_energy_mJ / duration if duration > 0 else 0
                
                # Peak power during inference
                peak_total_power = window_data['Power_mW'].max()
                peak_cpu_power = peak_total_power - avg_power_idle
                
                print(f"\nInference {i+1} ({duration*1000:.2f} ms, {len(window_data)} samples):")
                print(f"  Total Energy: {total_energy_mJ:.3f} mJ")
                print(f"  Idle Energy: {idle_energy_mJ:.3f} mJ")
                print(f"  CPU-only Energy: {cpu_energy_mJ:.3f} mJ")
                print(f"  CPU-only Avg Power: {cpu_avg_power_mW:.2f} mW")
                print(f"  CPU-only Peak Power: {peak_cpu_power:.2f} mW")
                print(f"  Peak/Avg Ratio: {peak_cpu_power/cpu_avg_power_mW:.2f}x" if cpu_avg_power_mW > 0 else "  Peak/Avg Ratio: N/A")
                
                total_cpu_energy += cpu_energy_mJ
                total_inference_time += duration
            
            if total_inference_time > 0:
                print(f"\n{'='*20} SUMMARY {'='*20}")
                print(f"Total CPU-only Energy: {total_cpu_energy:.3f} mJ")
                print(f"Total Inference Time: {total_inference_time*1000:.2f} ms")
                print(f"Average CPU-only Power: {total_cpu_energy/total_inference_time:.2f} mW")
                print(f"Energy per Inference: {total_cpu_energy/len(inference_windows):.3f} mJ")
    else:
        print("\nNo baseline power calculated - cannot compute CPU-only metrics")
        print("This could be due to:")
        print("- No inference windows defined")
        print("- Insufficient idle time between inferences")
        print("- All time periods overlap with inference windows")

if __name__ == "__main__":
    main()