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
    Detects available cameras, attempts to use a USB webcam,
    captures video, detects a single IR target, and saves its coordinates.
    Press 'q' or 'Ctrl/Cmd+C' to quit the application.
    """
    try:
        base_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
        file_path = get_new_file_path(base_path)

        available_cameras = CameraHandler.list_available_cameras()
        print(f"Available camera indices: {available_cameras}")

        if not available_cameras:
            print("No cameras detected. Please connect a camera and try again.")
            return

        camera_index = 1 if 1 in available_cameras else available_cameras[0]
        
        camera = CameraHandler(camera_index=camera_index)
        
        if not camera.open_camera():
            print(f"Failed to open camera with index {camera_index}. Please check the connection.")
            return

        print(f"Successfully opened camera with index {camera_index}. Press 'q' to quit.")
        print(f"Coordinates are being saved to: {file_path}")

        # Define min and max angles for mapping
        theta_min = -45.0  # Replace with actual values from the document
        theta_max = 45.0   # Replace with actual values from the document
        phi_min = 0.0      # Replace with actual values from the document
        phi_max = 90.0     # Replace with actual values from the document

        while True:
            frame = camera.get_frame()
            
            if frame is None:
                print("Failed to capture frame. Retrying...")
                time.sleep(1)
                continue

            gray_frame = apply_grayscale(frame)
            
            target = detect_single_target(gray_frame)
            
            if target:
                x, y, _ = target
                frame_with_target = draw_targets(frame, [target])
                mapped_x, mapped_y = map_coordinates(x, y, frame.shape[1], frame.shape[0], 1000, 1000)
                
                # Map pixel coordinates to spherical angles
                theta, phi = map_to_spherical_angles(x, y, frame.shape[1], frame.shape[0], theta_min, theta_max, phi_min, phi_max)
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                save_coordinates(file_path, mapped_x, mapped_y, timestamp)

                # Output the angles for further processing
                print(f"Mapped angles: θ = {theta:.2f}, ɸ = {phi:.2f}")
                
                cv2.imshow('IR Target Detection', frame_with_target)
                print(f"Target at ({x}, {y}) mapped to ({mapped_x:.2f}, {mapped_y:.2f}) with angles θ = {theta:.2f}, ɸ = {phi:.2f}")
            else:
                cv2.imshow('IR Target Detection', frame)
                print("No target detected")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting application...")
                break

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'camera' in locals():
            camera.release()
        cv2.destroyAllWindows()

# -----