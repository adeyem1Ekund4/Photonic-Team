# opencv-camera-project/srcs/utils/target_detection.py

import cv2
import numpy as np

def detect_ir_targets(frame, threshold=200, min_area=50, max_targets=5):
    """
    Detect IR LED or retro-reflector targets in a grayscale image.

    Parameters:
    frame (numpy.ndarray): Input grayscale image
    threshold (int): Brightness threshold for target detection (0-255)
    min_area (int): Minimum area of a target to be considered valid
    max_targets (int): Maximum number of targets to detect

    Returns:
    list: List of detected targets, each represented as (x, y, area)
    """
    # Ensure the frame is grayscale
    if len(frame.shape) > 2:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Threshold the image to isolate bright spots
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process contours to find centroids
    targets = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                targets.append((cX, cY, area))

    # Sort targets by area (largest first) and limit to max_targets
    targets.sort(key=lambda x: x[2], reverse=True)
    return targets[:max_targets]

def draw_targets(frame, targets, color=(0, 255, 0), radius=5, thickness=2):
    """
    Draw detected targets on the frame.

    Parameters:
    frame (numpy.ndarray): Input image (BGR)
    targets (list): List of targets, each represented as (x, y, area)
    color (tuple): BGR color for drawing
    radius (int): Radius of the circle to be drawn
    thickness (int): Thickness of the circle's line

    Returns:
    numpy.ndarray: Frame with targets drawn
    """
    for x, y, _ in targets:
        cv2.circle(frame, (x, y), radius, color, thickness)
        cv2.putText(frame, f"({x}, {y})", (x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame

def map_coordinates(x, y, frame_width, frame_height, out_width, out_height):
    """
    Map pixel coordinates to a different coordinate system.

    Parameters:
    x, y (int): Input pixel coordinates
    frame_width, frame_height (int): Dimensions of the input frame
    out_width, out_height (int): Dimensions of the output coordinate system

    Returns:
    tuple: Mapped (x, y) coordinates
    """
    mapped_x = (x / frame_width) * out_width
    mapped_y = (y / frame_height) * out_height
    return (mapped_x, mapped_y)
