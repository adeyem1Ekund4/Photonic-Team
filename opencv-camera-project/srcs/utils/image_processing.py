# opencv-camera-project/srcs/utils/image_processing.py
# This module contains utility functions for image processing.

import cv2

def apply_grayscale(frame):
    """
    Convert a given image frame to grayscale.

    Parameters:
    frame (numpy.ndarray): The input image frame in BGR format.

    Returns:
    numpy.ndarray: The grayscale image.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)