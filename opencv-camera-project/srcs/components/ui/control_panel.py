# opencv-camera-project/srcs/components/ui/control_panel.py
import cv2
import numpy as np
from utils.config import ConfigManager

class ControlPanel:
    def __init__(self, config_manager: ConfigManager):
        """
        Create a control panel with sliders for adjusting detection parameters.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.window_name = "Control Panel"
        self.panel_width = 400
        self.panel_height = 500
        self.is_visible = config_manager.get_display_config().get("show_control_panel", True)
        
        if self.is_visible:
            self._create_panel()
    
    def _create_panel(self):
        """Create the control panel window with trackbars."""
        # Create a window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.panel_width, self.panel_height)
        
        # Get current configuration
        detection_config = self.config_manager.get_detection_config()
        
        # Create trackbars for detection parameters
        cv2.createTrackbar("Threshold", self.window_name, 
                          detection_config.get("threshold_value", 245), 255, 
                          self._on_threshold_change)
        
        cv2.createTrackbar("Min Area", self.window_name, 
                          detection_config.get("min_area", 5), 100, 
                          self._on_min_area_change)
        
        cv2.createTrackbar("Max Area", self.window_name, 
                          min(detection_config.get("max_area", 500), 1000), 1000, 
                          self._on_max_area_change)
        
        # Convert float to int for trackbar (multiply by 100)
        circularity = int(detection_config.get("circularity_threshold", 0.7) * 100)
        cv2.createTrackbar("Circularity %", self.window_name, 
                          circularity, 100, 
                          self._on_circularity_change)
        
        # Convert float to int for trackbar (multiply by 100)
        tolerance = int(detection_config.get("square_tolerance", 0.2) * 100)
        cv2.createTrackbar("Square Tolerance %", self.window_name, 
                          tolerance, 100, 
                          self._on_tolerance_change)
        
        # Create method selector
        methods = ["brightness", "contour", "hybrid"]
        current_method = detection_config.get("method", "hybrid")
        method_index = methods.index(current_method) if current_method in methods else 0
        cv2.createTrackbar("Method", self.window_name, 
                          method_index, len(methods)-1, 
                          lambda x: self._on_method_change(x, methods))
        
        # Create a background image for the panel
        self.panel_image = np.ones((self.panel_height, self.panel_width, 3), dtype=np.uint8) * 240
        self._update_panel_image()
    
    def _on_threshold_change(self, value):
        """Callback for threshold trackbar."""
        detection_config = self.config_manager.get_detection_config()
        detection_config["threshold_value"] = value
        self.config_manager.update_section("detection", detection_config)
        self._update_panel_image()
    
    def _on_min_area_change(self, value):
        """Callback for min area trackbar."""
        detection_config = self.config_manager.get_detection_config()
        detection_config["min_area"] = value
        self.config_manager.update_section("detection", detection_config)
        self._update_panel_image()
    
    def _on_max_area_change(self, value):
        """Callback for max area trackbar."""
        detection_config = self.config_manager.get_detection_config()
        detection_config["max_area"] = value
        self.config_manager.update_section("detection", detection_config)
        self._update_panel_image()
    
    def _on_circularity_change(self, value):
        """Callback for circularity trackbar."""
        detection_config = self.config_manager.get_detection_config()
        # Convert from int to float (divide by 100)
        detection_config["circularity_threshold"] = value / 100.0
        self.config_manager.update_section("detection", detection_config)
        self._update_panel_image()
    
    def _on_tolerance_change(self, value):
        """Callback for square tolerance trackbar."""
        detection_config = self.config_manager.get_detection_config()
        # Convert from int to float (divide by 100)
        detection_config["square_tolerance"] = value / 100.0
        self.config_manager.update_section("detection", detection_config)
        self._update_panel_image()
    
    def _on_method_change(self, value, methods):
        """Callback for method selector trackbar."""
        detection_config = self.config_manager.get_detection_config()
        detection_config["method"] = methods[value]
        self.config_manager.update_section("detection", detection_config)
        self._update_panel_image()
    
    def _update_panel_image(self):
        """Update the panel image with current settings."""
        if not self.is_visible:
            return
            
        # Clear the image
        self.panel_image.fill(240)
        
        # Get current configuration
        detection_config = self.config_manager.get_detection_config()
        
        # Add title
        cv2.putText(self.panel_image, "Detection Settings", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add current values
        y_pos = 70
        line_height = 30
        
        # Method
        method = detection_config.get("method", "hybrid")
        cv2.putText(self.panel_image, f"Method: {method}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 1)
        y_pos += line_height
        
        # Threshold
        threshold = detection_config.get("threshold_value", 245)
        cv2.putText(self.panel_image, f"Threshold: {threshold}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 1)
        y_pos += line_height
        
        # Min Area
        min_area = detection_config.get("min_area", 5)
        cv2.putText(self.panel_image, f"Min Area: {min_area}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 1)
        y_pos += line_height
        
        # Max Area
        max_area = detection_config.get("max_area", 500)
        cv2.putText(self.panel_image, f"Max Area: {max_area}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 1)
        y_pos += line_height
        
        # Circularity
        circularity = detection_config.get("circularity_threshold", 0.7)
        cv2.putText(self.panel_image, f"Circularity: {circularity:.2f}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 1)
        y_pos += line_height
        
        # Square Tolerance
        tolerance = detection_config.get("square_tolerance", 0.2)
        cv2.putText(self.panel_image, f"Square Tolerance: {tolerance:.2f}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 1)
        y_pos += line_height * 2
        
        # Instructions
        cv2.putText(self.panel_image, "Instructions:", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += line_height
        
        instructions = [
            "- Adjust sliders to tune detection",
            "- Method: brightness, contour, or hybrid",
            "- Higher threshold = brighter dots only",
            "- Lower tolerance = stricter square shape",
            "- Press 'h' to hide/show this panel",
            "- Press 'r' to reset to defaults",
            "- Press 'q' to quit"
        ]
        
        for instruction in instructions:
            cv2.putText(self.panel_image, instruction, 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
            y_pos += line_height
        
        # Display the panel image
        cv2.imshow(self.window_name, self.panel_image)
    
    def update(self, key=None):
        """
        Update the control panel based on key presses.
        
        Args:
            key: Key code from cv2.waitKey
        """
        if not self.is_visible:
            # If panel is hidden and 'h' is pressed, show it
            if key == ord('h'):
                self.is_visible = True
                self._create_panel()
            return
        
        # Handle key presses
        if key == ord('h'):
            # Hide the panel
            self.is_visible = False
            cv2.destroyWindow(self.window_name)
        
        elif key == ord('r'):
            # Reset to defaults
            detection_config = self.config_manager.DEFAULT_CONFIG["detection"]
            self.config_manager.update_section("detection", detection_config)
            
            # Update trackbars to match defaults
            cv2.setTrackbarPos("Threshold", self.window_name, detection_config["threshold_value"])
            cv2.setTrackbarPos("Min Area", self.window_name, detection_config["min_area"])
            cv2.setTrackbarPos("Max Area", self.window_name, 
                              min(detection_config["max_area"], 1000))
            cv2.setTrackbarPos("Circularity %", self.window_name, 
                              int(detection_config["circularity_threshold"] * 100))
            cv2.setTrackbarPos("Square Tolerance %", self.window_name, 
                              int(detection_config["square_tolerance"] * 100))
            
            methods = ["brightness", "contour", "hybrid"]
            method_index = methods.index(detection_config["method"]) if detection_config["method"] in methods else 0
            cv2.setTrackbarPos("Method", self.window_name, method_index)
            
            self._update_panel_image()
    
    def close(self):
        """Close the control panel window."""
        if self.is_visible:
            cv2.destroyWindow(self.window_name)
