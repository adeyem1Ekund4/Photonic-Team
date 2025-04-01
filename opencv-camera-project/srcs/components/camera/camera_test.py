# opencv-camera-project/srcs/components/camera/camera_test.py
import cv2
import time
import numpy as np
import sys
import os

# Add the project source directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from components.camera.camera_handler import CameraHandler
from components.camera.camera_selector import CameraSelector
from utils.target_detection import detect_targets, TargetTracker, draw_square
from utils.image_processing import apply_grayscale, resize_frame
from utils.config import ConfigManager
from utils.performance_monitor import PerformanceMonitor
from components.ui.control_panel import ControlPanel

def main():
    """
    Main function for camera testing with target detection.
    """
    print("Initializing Camera Tracking Application...")
    
    # Load configuration
    config_manager = ConfigManager()
    camera_config = config_manager.get_camera_config()
    detection_config = config_manager.get_detection_config()
    display_config = config_manager.get_display_config()
    
    # Initialize performance monitor
    performance_monitor = PerformanceMonitor()
    
    # Select camera
    camera_selector = CameraSelector()
    selected_camera = camera_selector.select_camera()
    
    if selected_camera is None:
        print("No camera selected. Exiting application.")
        return
    
    # Initialize camera with configuration
    try:
        camera = CameraHandler(
            camera_index=selected_camera,
            resolution=tuple(camera_config["resolution"]),
            fps=camera_config["fps"]
        )
    except Exception as e:
        print(f"Error initializing camera: {e}")
        print("Exiting application.")
        return
    
    # Initialize target tracker with configuration
    target_tracker = TargetTracker(
        frame_width=camera.actual_resolution[0],
        frame_height=camera.actual_resolution[1],
        history_length=detection_config["history_length"],
        tolerance=detection_config["square_tolerance"]
    )
    
    # Initialize control panel
    control_panel = ControlPanel(config_manager)
    
    print("Camera initialized. Press 'q' to quit.")
    print("Press 'h' to hide/show control panel.")
    print("Press 'r' to reset detection parameters to defaults.")
    print("NOTE: Make sure the camera window is in focus when pressing keys")
    
    # Flag to control the main loop
    running = True
    
    while running:
        try:
            # Start timing this frame
            frame_start_time = time.time()
            
            # Get frame from camera
            frame = camera.get_frame()
            if frame is None:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Resize frame if scale factor is not 1.0
            scale_factor = display_config["scale_factor"]
            if scale_factor != 1.0:
                frame = resize_frame(frame, scale=scale_factor)
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Get current detection configuration (may have been updated by control panel)
            detection_config = config_manager.get_detection_config()
            
            # Detect dots using the configured method
            detection_method = detection_config["method"]
            dots = detect_targets(frame, 
                                 detection_mode=detection_method, 
                                 config=detection_config)
            
            # Draw detected dots
            for dot in dots:
                x, y, area = dot
                cv2.circle(display_frame, (x, y), 5, (0, 255, 255), -1)
                cv2.putText(display_frame, f"A:{int(area)}", (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Display number of dots detected
            if display_config["show_detection_info"]:
                cv2.putText(display_frame, f"Dots: {len(dots)} | Method: {detection_method}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Try to find a square arrangement
            square_dots, square_center = target_tracker.detect_advanced_target(
                frame, 
                detection_mode=detection_method,
                config=detection_config
            )

            if square_dots:
                # Draw the detected square
                display_frame = draw_square(display_frame, square_dots, color=(0, 255, 0), thickness=2)

                # `square_center` now contains the center coordinates
                if display_config["show_detection_info"]:
                    cv2.putText(display_frame, "Square Detected!", (square_center[0] - 60, square_center[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Draw a crosshair at the center
                cv2.drawMarker(display_frame, square_center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            else:
                # If no square detected, try to show predicted position
                predicted_center = target_tracker.predict_target_position()
                if predicted_center:
                    # Draw predicted position with different color (yellow)
                    cv2.drawMarker(display_frame, predicted_center, (0, 255, 255), 
                                cv2.MARKER_CROSS, 20, 2)
                    if display_config["show_detection_info"]:
                        cv2.putText(display_frame, "Predicted Position", 
                                (predicted_center[0] - 80, predicted_center[1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw target trajectory
            trajectory = target_tracker.get_target_trajectory()
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    # Draw line with increasing intensity for more recent points
                    intensity = int(255 * (i / len(trajectory)))
                    cv2.line(display_frame, trajectory[i-1], trajectory[i], 
                            (0, intensity, 255-intensity), 2)
            
            # Calculate frame processing time and update performance monitor
            frame_time = time.time() - frame_start_time
            performance_monitor.update(frame_time)
            
            # Display FPS if enabled
            if display_config["show_fps"]:
                fps = performance_monitor.get_fps()
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, display_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add quit instructions to the display
            cv2.putText(display_frame, "Press 'q' to quit | 'h' for panel | 'r' to reset", 
                       (display_frame.shape[1] - 400, display_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show the display frame
            cv2.imshow("Camera Feed", display_frame)
            
            # Use a shorter wait time to improve key responsiveness
            key = cv2.waitKey(10) & 0xFF
            
            # Update control panel with key press
            control_panel.update(key)
            
            # Check for q key press (ASCII code 113)
            if key == ord('q'):
                print("User requested exit (q key pressed)")
                running = False
                break
                
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Exiting...")
            running = False
            break
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            time.sleep(1)
    
    # Cleanup
    print("Cleaning up resources...")
    camera.release()
    control_panel.close()
    cv2.destroyAllWindows()
    
    # Print final performance stats
    stats = performance_monitor.get_stats()
    print("\nPerformance Summary:")
    print(f"  Average FPS: {stats['overall_fps']:.2f}")
    print(f"  Total frames processed: {stats['total_frames']}")
    print(f"  Total runtime: {stats['total_runtime']:.2f} seconds")
    print(f"  Average frame processing time: {stats['avg_frame_time']*1000:.2f} ms")
    print("Application terminated successfully")

if __name__ == "__main__":
    main()
