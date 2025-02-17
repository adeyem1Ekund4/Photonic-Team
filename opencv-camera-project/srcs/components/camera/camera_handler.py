# camera_handler.py
# opencv-camera-project/srcs/components/camera/camera_handler.py
# This module provides a class to handle camera operations.

import cv2

class CameraHandler:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        print(f"Initializing Windows camera with index {camera_index}")

    def open_camera(self):
        """
        Attempt to open the camera using Windows-specific settings.
        """
        try:
            # Try DirectShow first (preferred for Windows)
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            # Set resolution (optional, adjust as needed)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not self.cap.isOpened():
                print(f"Failed to open camera {self.camera_index} with DirectShow")
                # Try without DirectShow as fallback
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    print("Failed to open camera with fallback method")
                    return False

            # Verify camera works by reading a test frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Camera opened but failed to read frame")
                self.cap.release()
                return False

            print(f"Successfully opened camera {self.camera_index}")
            return True

        except Exception as e:
            print(f"Error opening camera: {str(e)}")
            return False

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @staticmethod
    def list_available_cameras():
        """
        List available cameras on Windows.
        """
        available_cameras = []
        for i in range(5):  # Check first 5 indices
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(i)
                    cap.release()
            except:
                continue
        return available_cameras



# -----

