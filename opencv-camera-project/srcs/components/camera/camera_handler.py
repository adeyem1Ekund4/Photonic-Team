# opencv-camera-project/srcs/components/camera/camera_handler.py
# This module provides a class to handle camera operations.

import cv2

class CameraHandler:
    """
    A class to handle camera operations such as capturing frames and releasing the camera.
    """

    def __init__(self, camera_index=0):
        """
        Initialize the CameraHandler with a specified camera index.

        Parameters:
        camera_index (int): The index of the camera to use (0 for built-in, 1 for external).
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open camera with index {camera_index}")

    def get_frame(self):
        """
        Capture a frame from the camera.

        Returns:
        numpy.ndarray or None: The captured frame, or None if capturing failed.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """
        Release the camera resource.
        """
        self.cap.release()

