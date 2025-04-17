# utils/green_corner_detection.py
import cv2
import numpy as np
from itertools import combinations

OPTIMAL_GREEN_PARAMS = {
    "hue_min": 39,          # Based on your feedback for better detection at distance
    "hue_max": 84,          # Based on your feedback for better detection at distance
    "sat_min": 18,          # Lowered substantially for distant/faded green markers
    "val_min": 30,          # Compromise value that works well at distance
    "qualityLevel": 0.01,   # Good balance for corner detection
    "minDistance": 5,       # Reduced to detect closer corners
    "maxCorners": 100,
    "morphIterations": 1,   # Keep minimal to avoid merging close corners
    "square_tolerance": 0.35, # Slightly increased to handle perspective distortion
    "min_area": 3,          # Reduced to detect smaller markers when distant
    "max_area": 800,        # Increased to handle larger markers when close
    "processing_scale": 0.75, # Process at 75% resolution for speed
    "history_length": 5     # Good balance for smooth tracking
}

def detect_green_regions_hsv(frame, config=None):
    if config is None:
        config = {}       
    # Get parameters from config with clear defaults and validation
    hue_min = max(0, min(179, config.get("hue_min", 40)))
    hue_max = max(0, min(179, config.get("hue_max", 80)))
    sat_min = max(1, min(255, config.get("sat_min", 50)))
    val_min = max(1, min(255, config.get("val_min", 50)))
    morph_iterations = max(0, min(10, config.get("morphIterations", 1)))   
    # Create HSV range for green detection
    lower_green = np.array([hue_min, sat_min, val_min])
    upper_green = np.array([hue_max, 255, 255])   
    # Convert to HSV and create mask - this is a performance-intensive operation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)   
    # Apply morphological operations to clean up the mask
    if morph_iterations > 0:  # Skip if iterations = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)   
    return mask

def detect_corners_in_mask(mask, config=None):
    if config is None:
        config = {}   
    # More efficient contour finding
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
    # Filter contours by area to remove noise
    min_area = config.get("min_area", 3)
    max_area = config.get("max_area", 800)   
    corners = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Fast centroid calculation
            M = cv2.moments(contour)
            if M["m00"] > 0:  # Avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                corners.append((cX, cY))   
    return corners

def rank_and_select_quad(corners, config=None):
    if config is None:
        config = {}
        
    if len(corners) < 4:
        return None      
    # Limit the number of corners to consider to avoid combinatorial explosion
    max_corners_to_check = min(20, len(corners))
    if len(corners) > max_corners_to_check:
        # Sort corners by distance from center of image to prioritize central corners
        h, w = 240, 320  # Approximate center, will work for most camera resolutions
        corners = sorted(corners, key=lambda pt: abs(pt[0] - w) + abs(pt[1] - h))
        corners = corners[:max_corners_to_check]
    
    best_quad = None
    max_area = 0  
    # Try all combinations of 4 corners
    for quad in combinations(corners, 4):
        pts = np.array(quad, dtype=np.float32)
        
        # Check if these points form a convex quadrilateral
        hull = cv2.convexHull(pts)
        if len(hull) == 4:  # It's a quadrilateral
            # Calculate area
            area = cv2.contourArea(hull)       
            # Use a more relaxed square check with higher tolerance
            tolerance = config.get("square_tolerance", 0.3)
            if is_square_like(quad, tolerance=tolerance):
                if area > max_area:
                    max_area = area
                    best_quad = quad  
    return best_quad

def is_square_like(quad, tolerance=0.3):
    # Calculate all pairwise distances
    pts = np.array(quad)
    dists = []
    for i in range(4):
        for j in range(i+1, 4):
            dist = np.linalg.norm(pts[i] - pts[j])
            dists.append(dist)    
    # Sort distances - in a square, 4 should be sides and 2 should be diagonals
    dists.sort()
    sides = dists[:4]
    diagonals = dists[4:]    
    # Check if sides are roughly equal
    avg_side = sum(sides) / 4
    side_deviation = max(abs(s - avg_side) for s in sides) / avg_side    
    # Check if diagonals are roughly equal
    if len(diagonals) == 2:
        diag_deviation = abs(diagonals[0] - diagonals[1]) / max(diagonals)
    else:
        diag_deviation = 1.0  # Not a quadrilateral    
    return side_deviation < tolerance and diag_deviation < tolerance

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

def auto_detect_green_hsv(frame, sample_regions=5):
    """Auto-detect HSV values for green markers with better error handling."""
    try:
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)      
        # Define a broad green range to start with
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])     
        # Create a mask for the broad green range
        broad_mask = cv2.inRange(hsv, lower_green, upper_green)      
        # Find contours in the broad mask
        contours, _ = cv2.findContours(broad_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)      
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # If no significant green regions found
        if not contours or cv2.contourArea(contours[0]) < 100:
            return None       
        # Sample HSV values from the largest green regions
        hue_values = []
        sat_values = []
        val_values = []
        
        for i in range(min(sample_regions, len(contours))):
            if cv2.contourArea(contours[i]) < 50:  # Skip very small contours
                continue
                
            # Create a mask for this contour
            contour_mask = np.zeros_like(broad_mask)
            cv2.drawContours(contour_mask, [contours[i]], 0, 255, -1)       
            # Get the average HSV values within this contour
            mean_hsv = cv2.mean(hsv, mask=contour_mask)     
            hue_values.append(mean_hsv[0])
            sat_values.append(mean_hsv[1])
            val_values.append(mean_hsv[2])
        
        if not hue_values:  # If no suitable contours were found
            return None
        
        # Calculate optimal HSV ranges based on samples
        mean_hue = np.mean(hue_values)
        hue_std = max(5.0, np.std(hue_values))  # Minimum std of 5 to ensure some range  
        # For saturation and value, we want minimums that capture the green objects
        mean_sat = np.mean(sat_values)
        mean_val = np.mean(val_values)  
        # Create the HSV config with some margins
        hsv_config = {
            "hue_min": max(0, int(mean_hue - hue_std * 1.5)),
            "hue_max": min(179, int(mean_hue + hue_std * 1.5)),
            "sat_min": max(10, int(mean_sat * 0.7)),  # 70% of mean saturation as minimum
            "val_min": max(10, int(mean_val * 0.7))   # 70% of mean value as minimum
        }
        return hsv_config
        
    except Exception as e:
        print(f"Error in auto green detection: {e}")
        # Return a fallback configuration
        return {
            "hue_min": 40,
            "hue_max": 80,
            "sat_min": 50,
            "val_min": 50
        }

class CornerTracker:
    def __init__(self, history_length=5):
        self.corner_history = []
        self.history_length = history_length
        self.velocity = [(0,0), (0,0), (0,0), (0,0)]  # Track velocity for each corner
        self.last_update_time = time.time()
        
    def update(self, corners):
        if corners is not None:
            current_time = time.time()
            dt = current_time - self.last_update_time
            
            # Calculate velocity if we have previous corners
            if self.corner_history:
                last_corners = self.corner_history[-1]
                if len(last_corners) == 4 and len(corners) == 4:
                    for i in range(4):
                        # Calculate velocity in pixels per second
                        dx = (corners[i][0] - last_corners[i][0]) / max(dt, 0.001)
                        dy = (corners[i][1] - last_corners[i][1]) / max(dt, 0.001)
                        # Apply smoothing to velocity (80% old, 20% new)
                        self.velocity[i] = (
                            0.8 * self.velocity[i][0] + 0.2 * dx,
                            0.8 * self.velocity[i][1] + 0.2 * dy
                        )
            
            self.corner_history.append(corners)
            if len(self.corner_history) > self.history_length:
                self.corner_history.pop(0)
                
            self.last_update_time = current_time
                
    def get_smoothed_corners(self):
        if not self.corner_history:
            return None
            
        # Average the corner positions over history with prediction
        smoothed = []
        for i in range(4):  # Assuming 4 corners
            x_sum = sum(history[i][0] for history in self.corner_history if len(history) > i)
            y_sum = sum(history[i][1] for history in self.corner_history if len(history) > i)
            count = sum(1 for history in self.corner_history if len(history) > i)
            
            if count > 0:
                # Basic position from history
                base_x = x_sum/count
                base_y = y_sum/count
                
                # Add velocity-based prediction (predict 1/30 second ahead)
                pred_x = base_x + self.velocity[i][0] * 0.033  # Assuming 30fps
                pred_y = base_y + self.velocity[i][1] * 0.033
                
                # Weighted blend of history and prediction (70% history, 30% prediction)
                smoothed.append((0.7 * base_x + 0.3 * pred_x, 
                                0.7 * base_y + 0.3 * pred_y))
                
        return smoothed if len(smoothed) == 4 else None


def detect_green_corners(frame, config=None):
    if config is None:
        config = OPTIMAL_GREEN_PARAMS.copy()  # Use optimized defaults
    
    # Resize frame for faster processing if needed
    processing_scale = config.get("processing_scale", 0.75)
    orig_size = frame.shape[:2]
    if processing_scale != 1.0:
        frame = cv2.resize(frame, (0, 0), fx=processing_scale, fy=processing_scale)
    
    # Phase 1: Detect green regions using HSV - optimize by using smaller blur kernel
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create HSV range for green detection
    lower_green = np.array([config.get("hue_min", 45), 
                           config.get("sat_min", 40), 
                           config.get("val_min", 40)])
    upper_green = np.array([config.get("hue_max", 85), 255, 255])
    
    # Fast mask creation
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply minimal morphological operations
    if config.get("morphIterations", 1) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Smaller kernel
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 
                              iterations=config.get("morphIterations", 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                              iterations=config.get("morphIterations", 1))
    
    # Phase 2: Find corners more efficiently
    corners = detect_corners_in_mask(mask, config)
    
    # Phase 3: Select four corners that form the best quadrilateral
    quad = rank_and_select_quad(corners, config)
    
    # Scale back if we resized
    if processing_scale != 1.0 and quad is not None:
        scale_factor = 1.0 / processing_scale
        quad = [(int(pt[0] * scale_factor), int(pt[1] * scale_factor)) for pt in quad]
    
    # Create perspective view if we have a valid quad
    perspective_view = None
    if quad is not None:
        # Scale the frame back to original size if we resized
        if processing_scale != 1.0:
            frame = cv2.resize(frame, (orig_size[1], orig_size[0]))
        ordered_quad = order_points(quad)
        perspective_view = perspective_transform(frame, ordered_quad)
    
    return quad, perspective_view, mask

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
