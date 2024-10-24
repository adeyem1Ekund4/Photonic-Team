# opencv-camera-project/srcs/components/camera/camera_test.py

import sys
import os
import cv2
import time
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from components.camera.camera_handler import CameraHandler
from utils.image_processing import apply_grayscale
from utils.target_detection import detect_single_target, draw_targets, map_coordinates

def get_new_file_path(base_path):
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
    with open(file_path, 'a') as f:
        f.write(f"{x:.2f},{y:.2f},{timestamp}\n")

def main():
    """
    Main function to test the camera functionality and target detection.
    Detects available cameras, attempts to use a USB webcam,
    captures video, detects a single IR target, and saves its coordinates.
    Press 'q' to quit the application.
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
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                save_coordinates(file_path, mapped_x, mapped_y, timestamp)
                
                cv2.imshow('IR Target Detection', frame_with_target)
                print(f"Target at ({x}, {y}) mapped to ({mapped_x:.2f}, {mapped_y:.2f})")
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

if __name__ == "__main__":
    main()

