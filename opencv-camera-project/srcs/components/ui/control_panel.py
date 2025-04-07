# components/ui/control_panel.py
import cv2
import numpy as np
from utils.config import ConfigManager

class ControlPanel:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.window_name = "Control Panel"
        self.panel_width = 400
        self.panel_height = 500
        self.is_visible = config_manager.get_display_config().get("show_control_panel", True)
        
        if self.is_visible:
            self._create_panel()
    
    def _create_panel(self):
        """Create the control panel window with trackbars for green corner detection."""
        # Create a window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.panel_width, self.panel_height)
        
        # Get current configuration
        detection_config = self.config_manager.get_detection_config()
        
        # HSV Color Range
        cv2.createTrackbar("Hue Min", self.window_name, 
                          detection_config.get("hue_min", 40), 179, 
                          self._on_hue_min_change)
        
        cv2.createTrackbar("Hue Max", self.window_name, 
                          detection_config.get("hue_max", 80), 179, 
                          self._on_hue_max_change)
        
        cv2.createTrackbar("Sat Min", self.window_name, 
                          detection_config.get("sat_min", 50), 255, 
                          self._on_sat_min_change)
        
        cv2.createTrackbar("Val Min", self.window_name, 
                          detection_config.get("val_min", 50), 255, 
                          self._on_val_min_change)
        
        # Corner Detection Parameters
        cv2.createTrackbar("Quality Level (x100)", self.window_name, 
                          int(detection_config.get("qualityLevel", 1) * 100), 100, 
                          self._on_quality_level_change)
        
        cv2.createTrackbar("Min Distance", self.window_name, 
                          detection_config.get("minDistance", 10), 50, 
                          self._on_min_distance_change)
        
        cv2.createTrackbar("Max Corners", self.window_name, 
                          detection_config.get("maxCorners", 100), 200, 
                          self._on_max_corners_change)
        
        # Morphological Operations
        cv2.createTrackbar("Morph Iterations", self.window_name, 
                          detection_config.get("morphIterations", 1), 5, 
                          self._on_morph_iterations_change)
        
        # Create a background image for the panel
        self.panel_image = np.ones((self.panel_height, self.panel_width, 3), dtype=np.uint8) * 240
        self._update_panel_image()
    
    def _on_hue_min_change(self, value):
        try:
            detection_config = self.config_manager.get_detection_config()
            
            hue_max = detection_config.get("hue_max", 80)
            value = min(value, hue_max)
            detection_config["hue_min"] = value
            self.config_manager.update_section("detection", detection_config)

            cv2.setTrackbarPos("Hue Min", self.window_name, value)
            self._update_panel_image()
        except Exception as e:
            print(f"Error in hue_min change: {e}")

    def _on_hue_max_change(self, value):
        try:
            detection_config = self.config_manager.get_detection_config()
            
            hue_min = detection_config.get("hue_min", 40)
            value = max(value, hue_min)
            detection_config["hue_max"] = value
            self.config_manager.update_section("detection", detection_config)
            
            cv2.setTrackbarPos("Hue Max", self.window_name, value)
            self._update_panel_image()
        except Exception as e:
            print(f"Error in hue_max change: {e}")

    def _on_sat_min_change(self, value):
        try:
            detection_config = self.config_manager.get_detection_config()
            detection_config["sat_min"] = value
            self.config_manager.update_section("detection", detection_config)
            self._update_panel_image()
        except Exception as e:
            print(f"Error in sat_min change: {e}")

    def _on_val_min_change(self, value):
        try:
            detection_config = self.config_manager.get_detection_config()
            detection_config["val_min"] = value
            self.config_manager.update_section("detection", detection_config)
            self._update_panel_image()
        except Exception as e:
            print(f"Error in val_min change: {e}")

    def _on_quality_level_change(self, value):
        try:
            detection_config = self.config_manager.get_detection_config()
            detection_config["qualityLevel"] = value / 100.0
            self.config_manager.update_section("detection", detection_config)
            self._update_panel_image()
        except Exception as e:
            print(f"Error in quality_level change: {e}")

    def _on_min_distance_change(self, value):
        try:
            detection_config = self.config_manager.get_detection_config()
            detection_config["minDistance"] = value
            self.config_manager.update_section("detection", detection_config)
            self._update_panel_image()
        except Exception as e:
            print(f"Error in min_distance change: {e}")

    def _on_max_corners_change(self, value):
        try:
            detection_config = self.config_manager.get_detection_config()
            detection_config["maxCorners"] = value
            self.config_manager.update_section("detection", detection_config)
            self._update_panel_image()
        except Exception as e:
            print(f"Error in max_corners change: {e}")

    def _on_morph_iterations_change(self, value):
        try:
            detection_config = self.config_manager.get_detection_config()
            detection_config["morphIterations"] = value
            self.config_manager.update_section("detection", detection_config)
            self._update_panel_image()
        except Exception as e:
            print(f"Error in morph_iterations change: {e}")
    
    def _update_panel_image(self):
        """Update the panel image with current settings."""
        if not self.is_visible:
            return
            
        # Clear the image
        self.panel_image.fill(240)
        
        # Get current configuration
        detection_config = self.config_manager.get_detection_config()
        
        # Add title
        cv2.putText(self.panel_image, "Green Corner Detection Settings", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 2)
        
        # Add current values with descriptions
        y_pos = 70
        line_height = 30
        
        # HSV Range
        cv2.putText(self.panel_image, f"Hue Range: {detection_config.get('hue_min', 40)}-{detection_config.get('hue_max', 80)}", 
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)
        cv2.putText(self.panel_image, "(Adjust to match your green markers)", 
                (20, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        y_pos += line_height + 20
        
        cv2.putText(self.panel_image, f"Sat Min: {detection_config.get('sat_min', 50)}", 
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)
        cv2.putText(self.panel_image, "(Higher values filter out whitish colors)", 
                (20, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        y_pos += line_height + 20
        
        cv2.putText(self.panel_image, f"Val Min: {detection_config.get('val_min', 50)}", 
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)
        cv2.putText(self.panel_image, "(Higher values filter out dark areas)", 
                (20, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        y_pos += line_height + 20
        
        # Corner Detection
        cv2.putText(self.panel_image, f"Quality Level: {detection_config.get('qualityLevel', 0.01):.2f}", 
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)
        cv2.putText(self.panel_image, "(Lower values detect more corners)", 
                (20, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        y_pos += line_height + 20
        
        cv2.putText(self.panel_image, f"Min Distance: {detection_config.get('minDistance', 10)}", 
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)
        cv2.putText(self.panel_image, "(Minimum pixel distance between corners)", 
                (20, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        y_pos += line_height + 20
        
        cv2.putText(self.panel_image, f"Max Corners: {detection_config.get('maxCorners', 100)}", 
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)
        cv2.putText(self.panel_image, "(Maximum corners to detect)", 
                (20, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        y_pos += line_height + 20
        
        cv2.putText(self.panel_image, f"Morph Iterations: {detection_config.get('morphIterations', 1)}", 
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 1)
        cv2.putText(self.panel_image, "(Higher values clean noise but blur edges)", 
                (20, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        y_pos += line_height * 2
        
        # Instructions
        cv2.putText(self.panel_image, "Instructions:", 
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += line_height
        
        instructions = [
            "- Adjust Hue to match your green markers",
            "- Increase Sat Min to filter out white/gray",
            "- Adjust Val Min to filter out dark areas",
            "- Lower Quality Level to detect more corners",
            "- Increase Min Distance to separate corners",
            "- Press 'd' to toggle debug view",
            "- Press 'r' to reset to defaults",
            "- Press 'h' to hide this panel"
        ]
        
        for instruction in instructions:
            cv2.putText(self.panel_image, instruction, 
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
            y_pos += line_height - 5  # Slightly reduced spacing for instructions
        
        # Display the panel image
        cv2.imshow(self.window_name, self.panel_image)
    
    def update(self, key=None):
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
            detection_config = {
                "hue_min": 40,
                "hue_max": 80,
                "sat_min": 50,
                "val_min": 50,
                "qualityLevel": 0.01,
                "minDistance": 10,
                "maxCorners": 100,
                "morphIterations": 1,
                # Preserve other settings not related to green corner detection
                "method": self.config_manager.get_detection_config().get("method", "hybrid"),
                "min_area": self.config_manager.get_detection_config().get("min_area", 5),
                "max_area": self.config_manager.get_detection_config().get("max_area", 500),
                "threshold_value": self.config_manager.get_detection_config().get("threshold_value", 245),
                "square_tolerance": self.config_manager.get_detection_config().get("square_tolerance", 0.2),
                "history_length": self.config_manager.get_detection_config().get("history_length", 5)
            }
            
            self.config_manager.update_section("detection", detection_config)
            
            # Update trackbars to match defaults
            cv2.setTrackbarPos("Hue Min", self.window_name, detection_config["hue_min"])
            cv2.setTrackbarPos("Hue Max", self.window_name, detection_config["hue_max"])
            cv2.setTrackbarPos("Sat Min", self.window_name, detection_config["sat_min"])
            cv2.setTrackbarPos("Val Min", self.window_name, detection_config["val_min"])
            cv2.setTrackbarPos("Quality Level (x100)", self.window_name, int(detection_config["qualityLevel"] * 100))
            cv2.setTrackbarPos("Min Distance", self.window_name, detection_config["minDistance"])
            cv2.setTrackbarPos("Max Corners", self.window_name, detection_config["maxCorners"])
            cv2.setTrackbarPos("Morph Iterations", self.window_name, detection_config["morphIterations"])
            
            self._update_panel_image()
    
    def close(self):
        """Close the control panel window."""
        if self.is_visible:
            cv2.destroyWindow(self.window_name)
