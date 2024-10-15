# opencv-camera-project/tests/test_main.py
# This script is used for testing the main functionality of the camera project.

import sys
import os
import cv2

# Add the src directory to the Python path to allow importing modules from it.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../srcs')))

from components.camera.camera_handler import CameraHandler
from utils.image_processing import apply_grayscale

def main():
    """
    Main function to capture video from a camera, process each frame to grayscale,
    and display it in a window. Press 'q' to quit the application.
    """
    camera = CameraHandler(camera_index=1)  # Use 1 for external USB camera, 0 for built-in

    while True:
        frame = camera.get_frame()  # Capture a frame from the camera
        if frame is None:
            print("Failed to capture frame")
            break

        # Apply grayscale processing to the captured frame
        gray_frame = apply_grayscale(frame)

        # Display the resulting grayscale frame
        cv2.imshow('Frame', gray_frame)

        # Press 'q' to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # Release the camera resource
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()  # Execute the main function if the script is run directly
