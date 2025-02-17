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

from components.camera.camera_handler import CameraHandler
from utils.image_processing import apply_grayscale
from utils.target_detection import detect_green_dots, validate_square, draw_square

def main():
    """
    Main function to test the camera functionality and target detection.
    """
    print("Initializing Drone Security Camera...")
    
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

            # Detect green dots
            detected_dots = detect_green_dots(frame)

            # Validate if the dots form a square
            if validate_square(detected_dots):
                frame = draw_square(frame, detected_dots)

                # Display coordinates of the square's center
                center_x = sum(dot[0] for dot in detected_dots) // 4
                center_y = sum(dot[1] for dot in detected_dots) // 4
                cv2.putText(frame, f"({center_x}, {center_y})", (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Add stylized elements
            height, width = frame.shape[:2]
            
            # Draw tracking plane scales
            cv2.line(frame, (0, height - 50), (width, height - 50), (255, 255, 255), 1)
            cv2.line(frame, (50, 0), (50, height), (255, 255, 255), 1)
            
            # Add scale markers
            for i in range(0, width, 50):
                cv2.line(frame, (i, height - 45), (i, height - 50), (255, 255, 255), 1)
                if i % 100 == 0:
                    cv2.putText(frame, str(i), (i-10, height - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            for i in range(0, height, 50):
                cv2.line(frame, (45, i), (50, i), (255, 255, 255), 1)
                if i % 100 == 0:
                    cv2.putText(frame, str(i), (20, i+5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Add corner LEDs
            led_radius = 5
            cv2.circle(frame, (10, 10), led_radius, (0, 255, 0), -1)  # Top-left
            cv2.circle(frame, (width-10, 10), led_radius, (0, 255, 0), -1)  # Top-right
            cv2.circle(frame, (10, height-10), led_radius, (0, 255, 0), -1)  # Bottom-left
            cv2.circle(frame, (width-10, height-10), led_radius, (0, 255, 0), -1)  # Bottom-right

            # Add center box LED indicators
            box_size = 20
            cv2.rectangle(frame, (width//2-box_size, height//2-box_size), 
                         (width//2+box_size, height//2+box_size), (0, 255, 0), 1)

            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (width - 150, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Display the frame
            cv2.imshow("Drone Security Camera", frame)

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