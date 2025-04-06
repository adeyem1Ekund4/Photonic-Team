# utils/green_corner_detection.py
import cv2
import numpy as np
from itertools import combinations


def detect_green_regions_hsv(frame, lower_green=np.array([40, 50, 50]), upper_green=np.array([80, 255, 255])):
    """
    Detect green regions in the frame via HSV segmentation.
    
    Args:
        frame (numpy.ndarray): Input image in BGR format.
        lower_green (numpy.ndarray): Lower bound for HSV green.
        upper_green (numpy.ndarray): Upper bound for HSV green.
        
    Returns:
        mask (numpy.ndarray): Binary mask of green regions.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def detect_corners_in_mask(mask, maxCorners=100, qualityLevel=0.01, minDistance=10):
    """
    Detect corner features within the green mask using Shi-Tomasi corner detection.
    
    Args:
        mask (numpy.ndarray): Binary mask from the green segmentation.
        maxCorners (int): Maximum number of corners to return.
        qualityLevel (float): Parameter characterizing the minimal accepted quality.
        minDistance (int): Minimum possible Euclidean distance between the returned corners.
        
    Returns:
        List of corner points (x, y) as tuples.
    """
    corners = cv2.goodFeaturesToTrack(mask, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance)
    if corners is not None:
        corners = np.int0(corners)
        return [tuple(c.ravel()) for c in corners]
    return []

def rank_and_select_quad(corners):
    """
    From candidate corner points, select four that form the best quadrilateral based on area.
    This brute-force approach cycles through all combinations of four points and returns the combination
    whose convex hull has the largest area.
    
    Args:
        corners (list): List of candidate corner points.
        
    Returns:
        best_quad (tuple): A tuple of 4 points (each a (x, y) tuple) or None if not found.
    """
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
    """
    Order four points in a consistent order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts (list or numpy.ndarray): List of four (x, y) points.
        
    Returns:
        ordered (numpy.ndarray): Array of ordered points.
    """
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
    """
    Apply a perspective transformation to the detected quadrilateral area to yield a normalized square view.
    
    Args:
        frame (numpy.ndarray): Original image.
        quad (list): List of four ordered corner points.
        
    Returns:
        warped (numpy.ndarray): Warped (normalized) view of the quadrilateral.
    """
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

def detect_green_corners(frame):
    """
    Full pipeline for green corner detection on a box with green tape markers. Steps include:
      1. Green region segmentation using HSV.
      2. Corner detection (using Shi-Tomasi) on the segmented mask.
      3. Ranking to select the best four candidate corners forming a quadrilateral.
      4. Optionally, applying a perspective transformation.
    
    Args:
        frame (numpy.ndarray): Input image in BGR format.
        
    Returns:
        quad (tuple): The four corner points of the detected quadrilateral (or None if not found).
        perspective_view (numpy.ndarray): The warped image of the quadrilateral area (or None).
    """
    # Phase 1: Detect green regions.
    mask = detect_green_regions_hsv(frame)
    
    # Phase 2: Detect candidate corner points within the green regions.
    corners = detect_corners_in_mask(mask)
    
    # Phase 3: Select four corners that form the best quadrilateral.
    quad = rank_and_select_quad(corners)
    
    perspective_view = None
    if quad is not None:
        ordered_quad = order_points(quad)
        perspective_view = perspective_transform(frame, ordered_quad)
    return quad, perspective_view

# The module can be tested independently:
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        quad, transformed = detect_green_corners(frame)
        if quad is not None:
            # Draw corner points
            for point in quad:
                cv2.circle(frame, tuple(map(int, point)), 5, (0, 255, 0), -1)
            
            # Draw lines between the points (using the ordered quadrilateral)
            pts = order_points(quad)
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1)%4]), (0, 0, 255), 2)
            cv2.putText(frame, "Square Detected", (int(pts[0][0]), int(pts[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Frame", frame)
        if transformed is not None:
            cv2.imshow("Perspective View", transformed)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
