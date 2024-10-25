# opencv-camera-project/srcs/components/camera/spherical_test.py

import sys
import os
import cv2
import time

# Add the srcs directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from components.camera.camera_handler import CameraHandler
from utils.image_processing import apply_grayscale
from utils.target_detection import detect_single_target, pixel_to_spherical

def main():
    """
    Main function to test the camera functionality and spherical coordinate conversion.
    Captures video, detects a single IR target, and prints its spherical coordinates.
    Press 'q' or 'Ctrl/Cmd+C' to quit the application.
    """
    try:
        camera = CameraHandler(camera_index=1)  # Use appropriate camera index
        if not camera.open_camera():
            print("Failed to open camera. Please check the connection.")
            return

        print("Camera opened successfully. Press 'q' to quit.")

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
                theta, phi = pixel_to_spherical(x, y, frame.shape[1], frame.shape[0])
                print(f"Target detected at pixel ({x}, {y}) -> Spherical coordinates (θ: {theta:.2f}, ϕ: {phi:.2f})")
            else:
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
