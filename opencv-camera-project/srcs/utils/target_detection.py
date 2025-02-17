# target_detection.py
# opencv-camera-project/srcs/utils/target_detection.py
# This module contains functions for detecting and processing targets in images.

import cv2
import numpy as np
from math import sqrt

def detect_green_dots(frame, min_area=50):
    """
    Detect bright green dots in the frame.

    Parameters:
    frame (numpy.ndarray): Input BGR image.
    min_area (int): Minimum area of a dot to be considered valid.

    Returns:
    list: List of detected dots as (x, y, area).
    """
    # Convert the frame to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for bright neon green color
    lower_green = np.array([40, 200, 200])  # Adjust these values for neon green
    upper_green = np.array([70, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_dots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                detected_dots.append((cX, cY, area))

    return detected_dots

def validate_square(dots, tolerance=0.2):
    """
    Validate if the detected dots form a square.

    Parameters:
    dots (list): List of detected dots as (x, y, area).
    tolerance (float): Allowed deviation from a perfect square.

    Returns:
    bool: True if the dots form a square, False otherwise.
    """
    if len(dots) != 4:
        return False

    # Calculate pairwise distances between all dots
    distances = []
    for i in range(len(dots)):
        for j in range(i + 1, len(dots)):
            x1, y1, _ = dots[i]
            x2, y2, _ = dots[j]
            distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distances.append(distance)

    # Sort distances and group them into sides and diagonals
    distances.sort()
    side = distances[:4]
    diagonal = distances[4:]

    # Check if the sides are approximately equal and diagonals are approximately equal
    side_avg = sum(side) / len(side)
    diagonal_avg = sum(diagonal) / len(diagonal)

    side_deviation = max(abs(s - side_avg) for s in side) / side_avg
    diagonal_deviation = max(abs(d - diagonal_avg) for d in diagonal) / diagonal_avg

    return side_deviation < tolerance and diagonal_deviation < tolerance


def draw_square(frame, dots, color=(0, 255, 0), thickness=2):
    """
    Draw a square around the detected dots.

    Parameters:
    frame (numpy.ndarray): Input image (BGR).
    dots (list): List of detected dots as (x, y, area).
    color (tuple): BGR color for drawing.
    thickness (int): Thickness of the lines.

    Returns:
    numpy.ndarray: Frame with the square drawn.
    """
    if len(dots) != 4:
        return frame

    # Sort dots by x-coordinate, then by y-coordinate
    dots = sorted(dots, key=lambda dot: (dot[0], dot[1]))

    # Draw lines between the dots
    for i in range(len(dots)):
        x1, y1, _ = dots[i]
        x2, y2, _ = dots[(i + 1) % len(dots)]
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

    return frame

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

def map_to_spherical_angles(x, y, frame_width, frame_height, theta_min, theta_max, phi_min, phi_max):
    """
    Map pixel coordinates to spherical angles (azimuth and altitude).

    Parameters:
    x, y (int): Input pixel coordinates
    frame_width, frame_height (int): Dimensions of the input frame
    theta_min, theta_max (float): Minimum and maximum azimuth angles
    phi_min, phi_max (float): Minimum and maximum altitude angles

    Returns:
    tuple: Mapped (theta, phi) angles
    """
    # Normalize the pixel coordinates to a range of 0 to 1
    normalized_x = x / frame_width
    normalized_y = y / frame_height

    # Map normalized coordinates to angle ranges
    theta = theta_min + (theta_max - theta_min) * normalized_x
    phi = phi_min + (phi_max - phi_min) * (1 - normalized_y)  # Invert y for altitude

    return (theta, phi)

# -----