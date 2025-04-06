# opencv-camera-project/srcs/utils/config.py
import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    DEFAULT_CONFIG = {
        "camera": {
            "default_index": 0,
            "resolution": [640, 480],
            "fps": 30,
            "brightness": 0,
            "contrast": 0,
            "saturation": 0,
            "auto_focus": True
        },
        "detection": {
            "method": "hybrid",  # "brightness", "contour", "hybrid"
            "min_area": 5,
            "max_area": 500,
            "threshold_value": 245,
            "square_tolerance": 0.2,
            "history_length": 5,
            "circularity_threshold": 0.7,
            "merge_distance": 10,
            "edge_detection": {
                "low_threshold": 50,
                "high_threshold": 150
            },
            # New parameters for green corner detection
            "hue_min": 40,
            "hue_max": 80,
            "sat_min": 50,
            "val_min": 50,
            "qualityLevel": 0.01,
            "minDistance": 10,
            "maxCorners": 100,
            "morphIterations": 1
        },
        "display": {
            "show_fps": True,
            "show_detection_info": True,
            "scale_factor": 1.0,
            "show_control_panel": True,
            "show_perspective": False
        },
        "save": {
            "save_detections": False,
            "output_directory": "output",
            "filename_prefix": "detection_"
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default if not exists.
        
        Returns:
            Dictionary containing configuration settings
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    return self._merge_with_defaults(config)
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
                return self.DEFAULT_CONFIG.copy()
        else:
            # Create default config file
            self.save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()
            
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge loaded config with defaults to ensure all keys exist.
        
        Args:
            config: Loaded configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = self.DEFAULT_CONFIG.copy()
        
        for section, values in config.items():
            if section in result:
                if isinstance(values, dict) and isinstance(result[section], dict):
                    # Merge section dictionaries
                    for key, value in values.items():
                        result[section][key] = value
                else:
                    # Replace entire section
                    result[section] = values
            else:
                # Add new section
                result[section] = values
                
        return result
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save (uses current config if None)
            
        Returns:
            True if successful, False otherwise
        """
        if config is None:
            config = self.config
            
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config file: {e}")
            return False
            
    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera-related configuration settings."""
        return self.config.get("camera", self.DEFAULT_CONFIG["camera"])
        
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection-related configuration settings."""
        return self.config.get("detection", self.DEFAULT_CONFIG["detection"])
        
    def get_display_config(self) -> Dict[str, Any]:
        """Get display-related configuration settings."""
        return self.config.get("display", self.DEFAULT_CONFIG["display"])
        
    def get_save_config(self) -> Dict[str, Any]:
        """Get saving-related configuration settings."""
        return self.config.get("save", self.DEFAULT_CONFIG["save"])
        
    def update_section(self, section: str, values: Dict[str, Any]) -> bool:
        """
        Update a configuration section with new values.
        
        Args:
            section: Section name to update
            values: New values to set
            
        Returns:
            True if successful, False otherwise
        """
        if section not in self.config:
            self.config[section] = {}
            
        for key, value in values.items():
            self.config[section][key] = value
            
        return self.save_config()
