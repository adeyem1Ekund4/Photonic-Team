# opencv-camera-project/srcs/utils/target_detection.py
import cv2
import numpy as np
import time
from itertools import combinations
from typing import List, Tuple, Optional

def detect_bright_dots(frame, min_area=5, threshold_value=245) -> List[Tuple[int, int, float]]:
    # Convert to HSV for better brightness isolation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    
    # Extract the V (Value) channel which represents brightness
    v_channel = hsv[:,:,2]  
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(v_channel, (5, 5), 0)   
    # Threshold the image to isolate very bright areas
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)   
    # Apply morphological operations to clean up the thresholded image
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    detected_dots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                detected_dots.append((cX, cY, area))   
    return detected_dots

def detect_targets_by_contour(frame, min_area=5, max_area=500, circularity_threshold=0.7) -> List[Tuple[int, int, float]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150) 
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    detected_dots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Check if the contour is approximately circular
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > circularity_threshold:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        detected_dots.append((cX, cY, area))
    return detected_dots

def merge_detections(bright_dots, contour_dots, max_distance=10) -> List[Tuple[int, int, float]]:
    merged_dots = list(bright_dots)  # Start with all brightness-detected dots   
    # Check each contour dot against brightness dots to avoid duplicates
    for c_dot in contour_dots:
        c_x, c_y, c_area = c_dot
        is_duplicate = False
        
        for b_dot in bright_dots:
            b_x, b_y, _ = b_dot
            # Calculate distance between dots
            distance = np.sqrt((c_x - b_x)**2 + (c_y - b_y)**2)
            if distance < max_distance:
                is_duplicate = True
                break      
        # If not a duplicate, add to merged list
        if not is_duplicate:
            merged_dots.append(c_dot)  
    return merged_dots

def detect_targets(frame, detection_mode="hybrid", config=None) -> List[Tuple[int, int, float]]:
    # Set default parameters if config not provided
    if config is None:
        config = {
            "min_area": 5,
            "max_area": 500,
            "threshold_value": 245,
            "circularity_threshold": 0.7
        }
    
    if detection_mode == "brightness":
        return detect_bright_dots(frame, 
                                  min_area=config.get("min_area", 5), 
                                  threshold_value=config.get("threshold_value", 245))    
    elif detection_mode == "contour":
        return detect_targets_by_contour(frame, 
                                        min_area=config.get("min_area", 5),
                                        max_area=config.get("max_area", 500),
                                        circularity_threshold=config.get("circularity_threshold", 0.7))   
    elif detection_mode == "hybrid":
        # Combine multiple detection methods for better accuracy
        bright_dots = detect_bright_dots(frame, 
                                        min_area=config.get("min_area", 5), 
                                        threshold_value=config.get("threshold_value", 245))       
        contour_dots = detect_targets_by_contour(frame, 
                                                min_area=config.get("min_area", 5),
                                                max_area=config.get("max_area", 500),
                                                circularity_threshold=config.get("circularity_threshold", 0.7))       
        return merge_detections(bright_dots, contour_dots, max_distance=config.get("merge_distance", 10))   
    else:
        # Default to brightness method
        return detect_bright_dots(frame, 
                                 min_area=config.get("min_area", 5), 
                                 threshold_value=config.get("threshold_value", 245))

def validate_square(dots: List[Tuple[int, int, float]], tolerance: float = 0.2) -> bool:
    if len(dots) != 4:
        return False
    # Calculate pairwise distances between all dots.
    distances = []
    for i in range(len(dots)):
        for j in range(i + 1, len(dots)):
            x1, y1, _ = dots[i]
            x2, y2, _ = dots[j]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distances.append(distance)
    distances.sort()
    # In a perfect square, the four smaller distances are the sides, and the two larger ones are the diagonals.
    sides = distances[:4]
    diagonals = distances[4:]
    side_avg = sum(sides) / len(sides)
    diag_avg = sum(diagonals) / len(diagonals)
    side_deviation = max(abs(s - side_avg) for s in sides) / side_avg
    diag_deviation = max(abs(d - diag_avg) for d in diagonals) / diag_avg
    return side_deviation < tolerance and diag_deviation < tolerance

def find_square_in_dots(dots: List[Tuple[int, int, float]], tolerance: float = 0.2) -> Optional[List[Tuple[int, int, float]]]:
    if len(dots) < 4:
        return None
    for combo in combinations(dots, 4):
        combo_list = list(combo)
        if validate_square(combo_list, tolerance):
            return combo_list
    return None

def draw_square(frame, dots: List[Tuple[int, int, float]], color=(0, 255, 0), thickness=2):
    if len(dots) != 4:
        return frame
    # Sort dots according to their positions: top-left, top-right, bottom-right, bottom-left.
    sorted_by_y = sorted(dots, key=lambda d: d[1])
    top_two = sorted(sorted_by_y[:2], key=lambda d: d[0])    # top-left, top-right
    bottom_two = sorted(sorted_by_y[2:], key=lambda d: d[0])   # bottom-left, bottom-right
    ordered = [top_two[0], top_two[1], bottom_two[1], bottom_two[0]]
    # Draw lines connecting the dots.
    for i in range(4):
        pt1 = (ordered[i][0], ordered[i][1])
        pt2 = (ordered[(i + 1) % 4][0], ordered[(i + 1) % 4][1])
        cv2.line(frame, pt1, pt2, color, thickness)
    return frame

class TargetTracker:
    def __init__(self, 
                 frame_width: int = 640, 
                 frame_height: int = 480,
                 history_length: int = 5,
                 tolerance: float = 0.2):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.history_length = history_length
        self.tolerance = tolerance     
        self.target_history = []
        self.last_detection_time = time.time()
        self.detection_timeout = 2.0  # seconds
        # Kalman Filter for smooth tracking (tracking using square center)
        self.kalman_filter = cv2.KalmanFilter(4, 2)
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                         [0, 1, 0, 0]], np.float32)
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                        [0, 1, 0, 1],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]], np.float32)
        self.kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                       [0, 1, 0, 0],
                                                       [0, 0, 1, 0],
                                                       [0, 0, 0, 1]], np.float32) * 0.03

    def _update_target_history(self, center: Tuple[int, int]):
        # Predict next state using Kalman filter.
        _ = self.kalman_filter.predict()
        measurement = np.array(center, dtype=np.float32)
        self.kalman_filter.correct(measurement)

        self.target_history.append(center)
        if len(self.target_history) > self.history_length:
            self.target_history.pop(0)
        self.last_detection_time = time.time()

    def predict_target_position(self) -> Optional[Tuple[int, int]]:
        if time.time() - self.last_detection_time > self.detection_timeout:
            return None    
        prediction = self.kalman_filter.predict()
        predicted_center = (int(prediction[0]), int(prediction[1]))
        return predicted_center

    def detect_advanced_target(self, frame: np.ndarray, detection_mode="hybrid", config=None) -> Tuple[Optional[List[Tuple[int, int, float]]], Optional[Tuple[int, int]]]:
        # Use the selected detection method
        dots = detect_targets(frame, detection_mode=detection_mode, config=config)
        square_dots = find_square_in_dots(dots, self.tolerance)
        
        if square_dots is not None:
            # Compute the center of the square to update the tracking history.
            xs = [pt[0] for pt in square_dots]
            ys = [pt[1] for pt in square_dots]
            center = (int(sum(xs) / 4), int(sum(ys) / 4))
            self._update_target_history(center)
            return square_dots, center  # Return both the corners and the center  
        # If no square is detected, try to predict position based on history
        if not square_dots and self.target_history:
            return None, None  # Still return None since we didn't actually detect the square     
        return None, None

    def get_target_trajectory(self) -> List[Tuple[int, int]]:
        return self.target_history
