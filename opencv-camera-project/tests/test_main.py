import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from components.camera.camera_handler import CameraHandler
from utils.image_processing import apply_grayscale

def main():
    camera = CameraHandler(camera_index=1)  # Use 1 for external USB camera, 0 for built-in

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Failed to capture frame")
            break

        # Apply some processing
        gray_frame = apply_grayscale(frame)

        # Display the resulting frame
        cv2.imshow('Frame', gray_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
