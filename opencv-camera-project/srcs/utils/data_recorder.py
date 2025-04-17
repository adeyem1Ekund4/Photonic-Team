# Add to utils/data_recorder.py
import os
import time
import csv
from datetime import datetime

class DataRecorder:
    def __init__(self, output_dir="output", filename_prefix="xy_greentarget_", interval_ms=250):
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.interval_ms = interval_ms
        self.last_record_time = 0
        self.file = None
        self.writer = None
        self.is_recording = False
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def start_recording(self):
        if self.is_recording:
            self.stop_recording()
            
        # Find the next available file number
        file_number = 1
        while True:
            filename = f"{self.filename_prefix}{file_number:03d}.txt"
            filepath = os.path.join(self.output_dir, filename)
            if not os.path.exists(filepath):
                break
            file_number += 1
        
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['timestamp', 'x', 'y'])
        self.is_recording = True
        self.last_record_time = time.time() * 1000  # Convert to ms
        print(f"Started recording to {filepath}")
        return filepath
    
    def record_point(self, center_point):
        if not self.is_recording or center_point is None:
            return False
            
        current_time = time.time() * 1000  # Convert to ms
        if current_time - self.last_record_time >= self.interval_ms:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.writer.writerow([timestamp, center_point[0], center_point[1]])
            self.file.flush()  # Ensure data is written immediately
            self.last_record_time = current_time
            return True
        return False
    
    def stop_recording(self):
        if self.is_recording and self.file:
            self.file.close()
            self.file = None
            self.writer = None
            self.is_recording = False
            print("Recording stopped")
            return True
        return False
