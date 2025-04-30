import serial
import time
import struct
import argparse
import pandas as pd
import numpy as np
import sys
import threading
import queue
from datetime import datetime
import tqdm
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, accuracy_score, 
    precision_score, recall_score
)
import numpy as np

def evaluate_predictions(y_true, y_pred, threshold=0.5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_pred_binary = (y_pred > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'auc': roc_auc_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary),
        'specificity': tn / (tn + fp + 1e-7),
        'f1_score': 2 * tp / (2 * tp + fp + fn + 1e-7),
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
    }

    return metrics




class PicoSerialInterface:
    def __init__(self, port, baud_rate=115200, timeout=1):
        """Initialize the serial connection to the Pico."""
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None
        self.stop_thread = False
        self.receive_queue = queue.Queue()
        self.receive_thread = None
        self.prediction_queue = queue.Queue()
        self.latency_queue = queue.Queue()
        
    def connect(self):
        """Establish serial connection."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            print(f"Connected to {self.port} at {self.baud_rate} baud")
            
            # Clear any pending data
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            # Start receive thread
            self.stop_thread = False
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            return True
        except serial.SerialException as e:
            print(f"Error connecting to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Close the serial connection."""
        self.stop_thread = True
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
        
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"Disconnected from {self.port}")
    


    

    def _receive_loop(self):
        buffer = b""
        while not self.stop_thread:
            if not self.ser or not self.ser.is_open:
                time.sleep(0.1)
                continue

            try:
                if self.ser.in_waiting > 0:
                    byte = self.ser.read(1)
                    if byte == b'\n':
                        try:
                            decoded = buffer.decode('utf-8').strip()
                            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            print(f"[{timestamp}] {decoded}")  # For logging

                            # Check for model output (e.g., "Result0.56" or "Result: 0.56")
                            if decoded.startswith("Result"):
                                self.prediction_queue.put(decoded)
                            if decoded.startswith("Latency"):
                                self.latency_queue.put(decoded)
                        except UnicodeDecodeError:
                            pass
                        buffer = b""
                    else:
                        buffer += byte
                else:
                    time.sleep(0.005)
            except Exception as e:
                print(f"Receive loop error: {e}")
                time.sleep(0.1)


    
    def get_received_data(self):
        """Get any received data from the queue (non-blocking)."""
        messages = []
        while not self.receive_queue.empty():
            messages.append(self.receive_queue.get_nowait())
        return messages
    
    def send_float_array(self, float_array):
        """Send an array of floats to the Pico."""
        if not self.ser or not self.ser.is_open:
            print("Serial port not open")
            return False
        
        try:

            float_array = np.asarray(float_array, dtype=np.float32)  # Ensure correct dtype
            floats_packed = struct.pack('<' + 'f' * len(float_array), *float_array)  # Pack all floats at once
            self.ser.write(floats_packed)
            self.ser.flush()
            print(f"Sent {len(float_array)} floats ({len(float_array) * 4} bytes)")
            return True
        except Exception as e:
            print(f"Error sending data: {e}")
            return False

def load_feather_data(file_path):
    """Load data from a feather file."""
    try:
        df = pd.read_feather(file_path)
        print(f"Loaded {len(df)} samples from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading feather file: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Send data to Pico over serial and receive results')
    parser.add_argument('port', help='Serial port (e.g., COM3 on Windows or /dev/ttyACM0 on Linux)')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--feather', help='Path to feather file with data to send')
    parser.add_argument('--timeout', type=float, default=1.0, help='Serial timeout in seconds')
    
    args = parser.parse_args()
    
    # Initialize serial interface
    pico = PicoSerialInterface(args.port, args.baud, args.timeout)
    
    if not pico.connect():
        sys.exit(1)
    
    try:
        # Wait for Pico to initialize
        print("Waiting for Pico to initialize...")
        time.sleep(2)
        
        # Display any startup messages
        messages = pico.get_received_data()
        for msg in messages:
            print(msg)
        
        # If feather file provided, load it
        df = None
        if args.feather:
            df = load_feather_data(args.feather)
            if df is None:
                sys.exit(1)
        
        # Main interaction loop
        positive_indices = df.index[df["Label"] == 1][:5]
        print(positive_indices)
        while True:
            # Check for any received messages
            #messages = pico.get_received_data()
            #for msg in messages:
            #    print(msg)
            
            # Command menu
            print("\nCommands:")
            print("1. Send test data (random)")
            print("2. Send sample from feather file")
            print("3. Enter float values manually")
            print("4. Run Complete Benchmark")
            print("5. Quit")
            
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == '1':
                # Generate random test data (60 values to match your model input)
                test_data = np.random.uniform(-1, 1, 60).astype(np.float32)
                pico.send_float_array(test_data)
            
            elif choice == '2':
                if df is None:
                    print("No feather file loaded. Use --feather option when starting the program.")
                    continue
                
                # Get sample index
                try:
                    sample_idx = int(input(f"Enter sample index (0-{len(df)-1}): "))
                    if sample_idx < 0 or sample_idx >= len(df):
                        print(f"Index must be between 0 and {len(df)-1}")
                        continue
                except ValueError:
                    print("Please enter a valid number")
                    continue
                
                # Send the sample
                sample = df['Segment'].iloc[sample_idx].astype(np.float32)
                pico.send_float_array(sample)
            
            elif choice == '3':
                # Manual entry
                try:
                    input_str = input("Enter comma-separated float values: ")
                    values = [float(x.strip()) for x in input_str.split(',')]
                    pico.send_float_array(np.array(values, dtype=np.float32))
                except ValueError:
                    print("Invalid input. Please enter valid comma-separated float values.")
            

            elif choice == '4':
                if df is None:
                    print("No feather file loaded. Use --feather option when starting the program.")
                    continue
                results = {}
                latency = 0.0
                # Choose range
                start_idx = 300
                end_idx = 600

                # Slice only the desired rows
                subset_df = df.iloc[start_idx:end_idx]
                for idx, row in tqdm.tqdm(subset_df.iterrows(), total=len(subset_df)):
                    sample = row['Segment'].astype(np.float32)
                    pico.send_float_array(sample)

                    try:
                        response = pico.prediction_queue.get(timeout=35) # Wait for result
                        lat = pico.latency_queue.get(timeout=35)
                        # Extract float from string like "Result0.56" or "Result: 0.56"
                        pred_str = response.split()[-1]  # or use regex
                        pred = float(pred_str)
                        lat = lat.split()[-1]
                        lat = int(lat)
                    except Exception as e:
                        print(f"No response for sample {idx}: {e}")
                        pred = None  # Handle error or retry

                    results[idx] = (row['Label'], pred)
                    latency += lat


                print("Benchmark Complete")
                y_true = [res[0] for res in results.values()]
                y_pred = [res[1] for res in results.values()]

                metrics = evaluate_predictions(y_true, y_pred)
                print(metrics)
                latency /= len(subset_df)
                print("Average Latency: " , latency)

            elif choice == '5':
                break
            
            else:
                print("Invalid choice")
            
            # Give time for Pico to process and respond
            time.sleep(0.5)
            
            # Display any responses
            messages = pico.get_received_data()
            for msg in messages:
                print(msg)
    
    finally:
        pico.disconnect()

if __name__ == "__main__":
    main()