#opencv-camera-project/srcs/components/camera/test_main.py

import sys
import os
import cv2
import time

# Add the parent directory of 'srcs' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from components.camera.camera_handler import CameraHandler
from utils.image_processing import apply_grayscale

def main():
    """
    Main function to test the camera functionality.
    Detects available cameras, attempts to use a USB webcam,
    captures video, converts each frame to grayscale,
    and displays it in a window. Press 'q' to quit the application.
    """
    try:
        # List available cameras
        available_cameras = CameraHandler.list_available_cameras()
        print(f"Available camera indices: {available_cameras}")

        if not available_cameras:
            print("No cameras detected. Please connect a camera and try again.")
            return

        # Prefer external USB camera (usually index 1) if available, else use the first available camera
        camera_index = 1 if 1 in available_cameras else available_cameras[0]
        
        camera = CameraHandler(camera_index=camera_index)
        
        if not camera.open_camera():
            print(f"Failed to open camera with index {camera_index}. Please check the connection.")
            return

        print(f"Successfully opened camera with index {camera_index}. Press 'q' to quit.")

        while True:
            frame = camera.get_frame()
            
            if frame is None:
                print("Failed to capture frame. Retrying...")
                time.sleep(1)  # Wait for a second before retrying
                continue

            gray_frame = apply_grayscale(frame)
            cv2.imshow('Camera Test', gray_frame)

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


