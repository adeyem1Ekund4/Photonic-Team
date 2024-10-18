#opencv-camera-project/srcs/components/camera/test_main.py

import sys
import os
import cv2

# Add the parent directory of 'srcs' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from components.camera.camera_handler import CameraHandler
from utils.image_processing import apply_grayscale

def main():
    """
    Main function to test the camera functionality.
    Captures video from the camera, converts each frame to grayscale,
    and displays it in a window. Press 'q' to quit the application.
    """
    try:
        # Initialize the camera (0 for built-in, 1 for external USB camera)
        camera = CameraHandler(camera_index=0)
        
        print("Camera initialized successfully. Press 'q' to quit.")

        while True:
            # Capture a frame from the camera
            frame = camera.get_frame()
            
            if frame is None:
                print("Failed to capture frame. Exiting...")
                break

            # Convert the frame to grayscale
            gray_frame = apply_grayscale(frame)

            # Display the grayscale frame
            cv2.imshow('Camera Test', gray_frame)

            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting application...")
                break

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'camera' in locals():
            camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




