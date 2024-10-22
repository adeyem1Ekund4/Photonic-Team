#opencv-camera-project/srcs/components/camera/test_main.py

import sys
import os
import cv2
import time

# Add the parent directory of 'srcs' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from components.camera.camera_handler import CameraHandler
from utils.image_processing import apply_grayscale
from utils.target_detection import detect_ir_targets, draw_targets, map_coordinates

def main():
    """
    Main function to test the camera functionality and target detection.
    Detects available cameras, attempts to use a USB webcam,
    captures video, detects IR targets, and displays the result.
    Press 'q' to quit the application.
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
            
            # Detect IR targets
            targets = detect_ir_targets(gray_frame)
            
            # Draw targets on the original frame
            frame_with_targets = draw_targets(frame, targets)
            
            # Display frame with detected targets
            cv2.imshow('IR Target Detection', frame_with_targets)

            # Print mapped coordinates (example mapping to 1000x1000 coordinate system)
            for x, y, _ in targets:
                mapped_x, mapped_y = map_coordinates(x, y, frame.shape[1], frame.shape[0], 1000, 1000)
                print(f"Target at ({x}, {y}) mapped to ({mapped_x:.2f}, {mapped_y:.2f})")

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


