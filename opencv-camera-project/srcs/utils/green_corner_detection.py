# utils/green_corner_detection.py
import cv2
import numpy as np
from itertools import combinations


def detect_green_regions_hsv(frame, config=None):
    """Detect green regions using HSV color space with configurable parameters."""
    if config is None:
        config = {}
    
    # Get parameters from config with clear defaults
    hue_min = config.get("hue_min", 40)  # Default green hue starts around 40
    hue_max = config.get("hue_max", 80)  # Default green hue ends around 80
    sat_min = config.get("sat_min", 50)  # Minimum saturation to filter out whitish colors
    val_min = config.get("val_min", 50)  # Minimum brightness to filter out dark areas
    morph_iterations = config.get("morphIterations", 1)  # Noise removal iterations
    
    # Create HSV range for green detection
    lower_green = np.array([hue_min, sat_min, val_min])
    upper_green = np.array([hue_max, 255, 255])
    
    # Convert to HSV and create mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    
    return mask

def detect_corners_in_mask(mask, config=None):
    """Detect corners in the mask with configurable parameters."""
    if config is None:
        config = {}

    max_corners = config.get("maxCorners", 100)  # Maximum number of corners to detect
    quality_level = config.get("qualityLevel", 0.01)  # Corner quality threshold (0-1)
    min_distance = config.get("minDistance", 10)  # Minimum distance between corners
    
    corners = cv2.goodFeaturesToTrack(mask, maxCorners=max_corners, 
                                     qualityLevel=quality_level, 
                                     minDistance=min_distance)
    if corners is not None:
        corners = np.int0(corners)
        return [tuple(c.ravel()) for c in corners]
    return []

def rank_and_select_quad(corners):
    if len(corners) < 4:
        return None
    best_quad = None
    max_area = 0
    for quad in combinations(corners, 4):
        pts = np.array(quad, dtype=np.float32)     
        hull = cv2.convexHull(pts)             
        if len(hull) == 4:       
            area = cv2.contourArea(hull)        
            if area > max_area:
                max_area = area
                best_quad = quad
    return best_quad

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    # The top-left will have the smallest sum and bottom-right the largest sum
    s = pts.sum(axis=1)
    # The top-right will have the smallest difference and bottom-left the largest difference
    diff = np.diff(pts, axis=1)
    
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def perspective_transform(frame, quad):
    ordered = order_points(quad)
    (tl, tr, br, bl) = ordered

    # Compute the width of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    # Compute the height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    # Destination points for perspective transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    return warped

class CornerTracker:
    def __init__(self, history_length=5):
        self.corner_history = []
        self.history_length = history_length
        
    def update(self, corners):
        if corners is not None:
            self.corner_history.append(corners)
            if len(self.corner_history) > self.history_length:
                self.corner_history.pop(0)
                
    def get_smoothed_corners(self):
        if not self.corner_history:
            return None
            
        # Average the corner positions over history
        smoothed = []
        for i in range(4):  # Assuming 4 corners
            x_sum = sum(history[i][0] for history in self.corner_history if len(history) > i)
            y_sum = sum(history[i][1] for history in self.corner_history if len(history) > i)
            count = sum(1 for history in self.corner_history if len(history) > i)
            
            if count > 0:
                smoothed.append((x_sum/count, y_sum/count))
                
        return smoothed if len(smoothed) == 4 else None

def detect_green_corners(frame, config=None):
    """Detect green corners in the frame with configurable parameters."""
    if config is None:
        config = {}
    
    # Phase 1: Detect green regions.
    mask = detect_green_regions_hsv(frame, config)
    
    # Phase 2: Detect candidate corner points within the green regions.
    corners = detect_corners_in_mask(mask, config)
    
    # Phase 3: Select four corners that form the best quadrilateral.
    quad = rank_and_select_quad(corners)
    
    perspective_view = None
    if quad is not None:
        ordered_quad = order_points(quad)
        perspective_view = perspective_transform(frame, ordered_quad)
    return quad, perspective_view, mask  # Also return mask for debugging

# The module can be tested independently:
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    # Create a simple window for adjusting parameters
    cv2.namedWindow("Controls")
    cv2.createTrackbar("Hue Min", "Controls", 40, 179, lambda x: None)
    cv2.createTrackbar("Hue Max", "Controls", 80, 179, lambda x: None)
    cv2.createTrackbar("Sat Min", "Controls", 50, 255, lambda x: None)
    cv2.createTrackbar("Val Min", "Controls", 50, 255, lambda x: None)
    cv2.createTrackbar("Quality Level", "Controls", 1, 100, lambda x: None)
    cv2.createTrackbar("Min Distance", "Controls", 10, 50, lambda x: None)
    cv2.createTrackbar("Morph Iterations", "Controls", 1, 5, lambda x: None)
    
    # Initialize corner tracker for smoothing
    corner_tracker = CornerTracker(history_length=5)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get parameters from trackbars
        config = {
            "hue_min": cv2.getTrackbarPos("Hue Min", "Controls"),
            "hue_max": cv2.getTrackbarPos("Hue Max", "Controls"),
            "sat_min": cv2.getTrackbarPos("Sat Min", "Controls"),
            "val_min": cv2.getTrackbarPos("Val Min", "Controls"),
            "qualityLevel": cv2.getTrackbarPos("Quality Level", "Controls") / 100.0,
            "minDistance": cv2.getTrackbarPos("Min Distance", "Controls"),
            "morphIterations": cv2.getTrackbarPos("Morph Iterations", "Controls")
        }
        
        # Detect green corners
        quad, transformed, mask = detect_green_corners(frame, config)
        
        # Update corner tracker
        if quad is not None:
            corner_tracker.update(quad)
            
        # Get smoothed corners
        smoothed_quad = corner_tracker.get_smoothed_corners()
        
        # Display mask for debugging
        cv2.imshow("Green Mask", mask)
        
        if smoothed_quad is not None:
            # Draw smoothed corner points
            for point in smoothed_quad:
                cv2.circle(frame, tuple(map(int, point)), 5, (255, 0, 0), -1)
            
            # Draw lines between the points (using the ordered quadrilateral)
            pts = order_points(smoothed_quad)
            for i in range(4):
                pt1 = tuple(map(int, pts[i]))
                pt2 = tuple(map(int, pts[(i+1)%4]))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
            
            cv2.putText(frame, "Square Detected", (int(pts[0][0]), int(pts[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # If original (non-smoothed) corners detected, also show them
        if quad is not None:
            for point in quad:
                cv2.circle(frame, tuple(map(int, point)), 3, (0, 255, 0), -1)
        
        cv2.imshow("Frame", frame)
        if transformed is not None:
            cv2.imshow("Perspective View", transformed)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()
