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
        """Background thread to continuously read from serial port."""
        while not self.stop_thread:
            if not self.ser or not self.ser.is_open:
                time.sleep(0.1)
                continue
                
            try:
                # Check if data is available
                if self.ser.in_waiting > 0:
                    line = self.ser.readline()
                    if line:
                        try:
                            decoded = line.decode('utf-8').strip()
                            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            self.receive_queue.put(f"[{timestamp}] {decoded}")
                        except UnicodeDecodeError:
                            # Handle binary data
                            self.receive_queue.put(f"[Binary data received: {len(line)} bytes]")
                else:
                    time.sleep(0.01)  # Short sleep to prevent CPU hogging
            except serial.SerialException as e:
                print(f"Serial error: {e}")
                break
            except Exception as e:
                print(f"Error in receive loop: {e}")
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
        while True:
            # Check for any received messages
            messages = pico.get_received_data()
            for msg in messages:
                print(msg)
            
            # Command menu
            print("\nCommands:")
            print("1. Send test data (random)")
            print("2. Send sample from feather file")
            print("3. Enter float values manually")
            print("4. Quit")
            
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