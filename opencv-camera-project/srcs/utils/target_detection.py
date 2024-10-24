# opencv-camera-project/srcs/utils/target_detection.py

import cv2
import numpy as np

def detect_single_target(frame, threshold=200, min_area=50):
    """
    Detect the brightest spot in a grayscale image, assumed to be the Photo Beam Sensor/Retroreflector.

    Parameters:
    frame (numpy.ndarray): Input grayscale image
    threshold (int): Brightness threshold for target detection (0-255)
    min_area (int): Minimum area of a target to be considered valid

    Returns:
    tuple: Detected target represented as (x, y, area), or None if no target found
    """
    # Ensure the frame is grayscale
    if len(frame.shape) > 2:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Threshold the image to isolate bright spots
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour above the minimum area
    largest_contour = None
    largest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area > largest_area:
            largest_contour = contour
            largest_area = area

    # If a valid contour is found, calculate its centroid
    if largest_contour is not None:
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY, largest_area)

    return None

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
