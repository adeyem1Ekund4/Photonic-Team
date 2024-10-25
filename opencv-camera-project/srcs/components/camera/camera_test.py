# opencv-camera-project/srcs/components/camera/camera_test.py

import sys
import os
import cv2
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from components.camera.camera_handler import CameraHandler
from utils.image_processing import apply_grayscale
from utils.target_detection import detect_single_target, draw_targets, map_coordinates

def get_new_file_path(base_path):
    file_index = 1
    while True:
        file_path = os.path.join(base_path, f'target_coordinates_{file_index:03d}.txt')
        if not os.path.exists(file_path):
            return file_path
        file_index += 1

class SphericalMapper:
    def __init__(self, frame_width, frame_height, fov_horizontal=90, fov_vertical=60):
        """
        Initialize the spherical coordinate mapper.
        
        Parameters:
        frame_width, frame_height: Camera resolution in pixels
        fov_horizontal: Horizontal field of view in degrees (default 90°)
        fov_vertical: Vertical field of view in degrees (default 60°)
        """
        self.width = frame_width
        self.height = frame_height
        self.fov_h = fov_horizontal
        self.fov_v = fov_vertical
        
        # Calculate angular ranges
        self.theta_min = 45  # degrees (left edge)
        self.theta_max = 135  # degrees (right edge)
        self.phi_min = 60  # degrees (bottom edge)
        self.phi_max = 120  # degrees (top edge)
        
        # Create coordinate grids
        self.setup_coordinate_grids()

    def setup_coordinate_grids(self):
        """Create coordinate grids for faster mapping"""
        x = np.linspace(0, self.width - 1, self.width)
        y = np.linspace(0, self.height - 1, self.height)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Pre-calculate the angular grids
        self.theta_grid = self.pixel_to_theta(self.X)
        self.phi_grid = self.pixel_to_phi(self.Y)

    def pixel_to_spherical(self, x, y):
        """
        Convert pixel coordinates to spherical coordinates (θ,ϕ)
        
        Parameters:
        x, y: Pixel coordinates
        
        Returns:
        tuple: (theta, phi) in degrees
        """
        theta = self.pixel_to_theta(x)
        phi = self.pixel_to_phi(y)
        return theta, phi

    def pixel_to_theta(self, x):
        """Convert x-pixel to azimuth angle θ"""
        # Apply barrel distortion
        x_normalized = x / self.width
        # Non-linear mapping for barrel distortion
        theta = self.theta_min + (self.theta_max - self.theta_min) * \
                (1 + np.sin((x_normalized - 0.5) * np.pi)) / 2
        return theta

    def pixel_to_phi(self, y):
        """Convert y-pixel to altitude angle ϕ"""
        # Invert y-axis (0 at bottom)
        y_normalized = 1 - (y / self.height)
        # Non-linear mapping for barrel distortion
        phi = self.phi_min + (self.phi_max - self.phi_min) * \
              (1 + np.sin((y_normalized - 0.5) * np.pi)) / 2
        return phi

    def draw_grid(self, frame, spacing=50):
        """
        Draw spherical coordinate grid on frame
        
        Parameters:
        frame: Input image
        spacing: Grid line spacing in pixels
        """
        # Draw latitude lines (constant phi)
        for y in range(0, self.height, spacing):
            phi = self.pixel_to_phi(y)
            cv2.line(frame, (0, y), (self.width, y), (0, 255, 0), 1)
            cv2.putText(frame, f"ϕ={phi:.1f}°", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw longitude lines (constant theta)
        for x in range(0, self.width, spacing):
            theta = self.pixel_to_theta(x)
            cv2.line(frame, (x, 0), (x, self.height), (0, 255, 0), 1)
            cv2.putText(frame, f"θ={theta:.1f}°", (x, self.height-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Update the main function in camera_test.py:

def main():
    try:
        base_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
        file_path = get_new_file_path(base_path)

        available_cameras = CameraHandler.list_available_cameras()
        print(f"Available camera indices: {available_cameras}")

        if not available_cameras:
            print("No cameras detected. Please connect a camera and try again.")
            return

        camera_index = 1 if 1 in available_cameras else available_cameras[0]
        camera = CameraHandler(camera_index=camera_index)
        
        if not camera.open_camera():
            print(f"Failed to open camera with index {camera_index}")
            return

        # Get initial frame to setup mapper
        frame = camera.get_frame()
        if frame is None:
            print("Failed to get initial frame")
            return

        # Initialize spherical mapper with frame dimensions
        mapper = SphericalMapper(frame.shape[1], frame.shape[0])

        print(f"Successfully opened camera with index {camera_index}. Press 'q' to quit.")
        print(f"Coordinates are being saved to: {file_path}")

        while True:
            frame = camera.get_frame()
            
            if frame is None:
                print("Failed to capture frame. Retrying...")
                time.sleep(1)
                continue

            gray_frame = apply_grayscale(frame)
            target = detect_single_target(gray_frame)
            
            # Draw the spherical coordinate grid
            mapper.draw_grid(frame)
            
            if target:
                x, y, _ = target
                frame_with_target = draw_targets(frame, [target])
                
                # Convert to spherical coordinates
                theta, phi = mapper.pixel_to_spherical(x, y)
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                
                # Save spherical coordinates instead of Cartesian
                save_coordinates(file_path, theta, phi, timestamp)
                
                cv2.imshow('IR Target Detection', frame_with_target)
                print(f"Target at pixel ({x}, {y}) mapped to (θ={theta:.2f}°, ϕ={phi:.2f}°)")
            else:
                cv2.imshow('IR Target Detection', frame)
                print("No target detected")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting application...")
                break

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'camera' in locals():
            camera.release()
        cv2.destroyAllWindows()

# Update the save_coordinates function to handle spherical coordinates:

def save_coordinates(file_path, theta, phi, timestamp):
    """
    Save the spherical coordinates to the specified file.
    
    Parameters:
    file_path (str): Path to the file where coordinates should be saved
    theta (float): Azimuth angle in degrees
    phi (float): Altitude angle in degrees
    timestamp (str): Timestamp for the coordinate capture
    """
    date, time = timestamp.split(' ')

    file_exists = os.path.exists(file_path)

    with open(file_path, 'a') as f:
        if not file_exists:
            f.write("theta(degrees),phi(degrees),date,time\n")
        f.write(f"{theta:.2f},{phi:.2f},{date},{time}\n")
