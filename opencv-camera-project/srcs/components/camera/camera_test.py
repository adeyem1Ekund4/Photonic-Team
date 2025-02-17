# camera_test.py
# opencv-camera-project/srcs/components/camera/camera_test.py
# This script tests the camera functionality and performs target detection.

import sys
import os
import cv2
import time
import datetime

# Add the project source directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from components.camera.camera_handler import CameraHandler
from utils.image_processing import apply_grayscale
from utils.target_detection import detect_single_target, draw_targets, map_coordinates, map_to_spherical_angles

def get_new_file_path(base_path):
    """
    Generate a new file path for saving target coordinates.

    Parameters:
    base_path (str): The base directory path.

    Returns:
    str: A new file path for saving target coordinates.
    """
    file_index = 1
    while True:
        file_path = os.path.join(base_path, f'target_coordinates_{file_index:03d}.txt')
        if not os.path.exists(file_path):
            return file_path
        file_index += 1

def save_coordinates(file_path, x, y, timestamp):
    """
    Save the mapped coordinates to the specified file.

    Parameters:
    file_path (str): Path to the file where coordinates should be saved.
    x, y (float): Mapped coordinates
    timestamp (str): Timestamp for the coordinate capture
    """
    date, time = timestamp.split(' ')

    file_exists = os.path.exists(file_path)

    with open(file_path, 'a') as f:
        # Write headers if the file is new
        if not file_exists:
            f.write("x-coordinate,y-coordinate,date,time\n")
        # Write the data
        f.write(f"{x:.2f},{y:.2f},{date},{time}\n")

def main():
    """
    Main function to test the camera functionality and target detection.
    """
    try:
        camera = CameraHandler(camera_index=0)  # Use the first available camera

        if not camera.open_camera():
            print("Failed to open camera. Please check the connection.")
            return

        print("Press 'q' to quit.")

        while True:
            frame = camera.get_frame()

            if frame is None:
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

            # Add stylized elements (e.g., XY tracking plane scales)
            height, width, _ = frame.shape
            cv2.line(frame, (0, height - 50), (width, height - 50), (255, 255, 255), 1)
            cv2.line(frame, (50, 0), (50, height), (255, 255, 255), 1)

            # Display the frame
            cv2.imshow("Drone Security Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'camera' in locals():
            camera.release()
        cv2.destroyAllWindows()

# -----