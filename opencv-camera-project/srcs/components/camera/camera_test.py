# camera_test.py
# opencv-camera-project/srcs/components/camera/camera_test.py
# This script tests the camera functionality and performs target detection.

# camera_test.py
import cv2
import time
import numpy as np
import sys
import os

# Add the project source directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.target_detection import TargetTracker
from utils.image_processing import apply_grayscale

def main():
    """
    Main function to test the camera functionality and advanced target detection.
    """
    print("Initializing Advanced Camera Tracking...")
    
    # Initialize TargetTracker
    target_tracker = TargetTracker()
    
    try:
        # Initialize camera with DirectShow backend (Windows)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("Failed to open camera. Please check your camera connection.")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Camera opened successfully. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Failed to capture frame. Retrying...")
                time.sleep(1)
                continue

            # Detect advanced target
            target = target_tracker.detect_advanced_target(frame)

            if target:
                x, y = target
                # Draw target
                cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
                cv2.putText(frame, f"Target: ({x}, {y})", (x+15, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Get and draw trajectory
            trajectory = target_tracker.get_target_trajectory()
            for i in range(1, len(trajectory)):
                cv2.line(frame, trajectory[i-1], trajectory[i], (255, 0, 0), 2)

            # Display the frame
            cv2.imshow("Advanced Target Tracking", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# -----