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
from utils.green_corner_detection import detect_green_corners, order_points, CornerTracker, detect_corners_in_mask
from utils.image_processing import apply_grayscale, resize_frame
from utils.config import ConfigManager
from utils.performance_monitor import PerformanceMonitor
from components.ui.control_panel import ControlPanel

def main():
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
    
    # Initialize control panel
    control_panel = ControlPanel(config_manager)   
    # Initialize corner tracker for temporal smoothing
    corner_tracker = CornerTracker(history_length=detection_config.get("history_length", 5))   
    # Add debugging flag
    debug_mode = False
    auto_green_mode = False
    print("Camera initialized. Press 'q' to quit.")
    print("Press 'h' to hide/show control panel.")
    print("Press 'r' to reset detection parameters to defaults.")
    print("Press 'p' to toggle perspective view.")
    print("Press 'm' to toggle mask view.")
    print("Press 'd' to toggle debug mode (helps with parameter tuning).")
    print("Press 'a' to toggle auto green detection mode.")
    print("NOTE: Make sure the camera window is in focus when pressing keys")
    print("\nTIPS FOR GREEN MARKER DETECTION:")
    print("1. Use bright green markers (post-it notes work well)")
    print("2. Ensure good lighting conditions")
    print("3. Adjust Hue Min/Max to match your specific shade of green")
    print("4. Increase Sat Min if detecting too many non-green objects")
    print("5. Use debug mode ('d' key) to see what's being detected")    
    # Flag to control the main loop
    running = True
    show_perspective = display_config.get("show_perspective", False)
    show_mask = False  # New flag to toggle mask display  
    # List to store recent center positions for trajectory tracking
    trajectory = []
    max_trajectory_length = detection_config.get("history_length", 5)
    
    while running:
        frame_count = 0 
        try:
            frame_count += 1
            if frame_count % 2 != 0 and config_manager.get_detection_config().get("enable_frame_skip", False):
                # Process only every other frame on low-power devices
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                    break
                continue
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
             # Create a separate copy for processing (could be even smaller for faster processing)
            processing_scale = config_manager.get_detection_config().get("processing_scale", 1.0)
            if processing_scale != 1.0 and processing_scale != scale_factor:
                processing_frame = resize_frame(frame.copy(), scale=processing_scale)
            else:
                processing_frame = frame.copy()          
            # Create a copy for display
            display_frame = frame.copy()        
            # Get current detection configuration (may have been updated by control panel)
            detection_config = config_manager.get_detection_config()
            # If auto green mode is active, automatically determine HSV values
            if auto_green_mode:
                # Auto-determine HSV values for green detection
                auto_hsv_values = auto_detect_green_hsv(frame)
                if auto_hsv_values:
                    # Update detection/auto-determined values
                    detection_config.update(auto_hsv_values)
                    # Update trackbars/visible
                    if control_panel.is_visible:
                        control_panel.update_trackbars_from_config(detection_config)

            # Detect green corners using the updated method
            try:
                quad, perspective_view, mask = detect_green_corners(processing_frame, detection_config)
                # If the processing frame was scaled, adjust the coordinates for the display frame
                if processing_scale != 1.0:
                    if quad is not None:
                        scale_ratio = 1.0 / processing_scale
                        quad = [(int(x * scale_ratio), int(y * scale_ratio)) for x, y in quad]
            
            except Exception as e:
                    print(f"Error in green corner detection: {e}")
                    # Reset to default detection parameters if an error occurs
                    detection_config = config_manager.DEFAULT_CONFIG["detection"]
                    config_manager.update_section("detection", detection_config)
                    control_panel.update_trackbars_from_config(detection_config)
                    # Skip this frame
                    continue
            
            # Show mask if enabled or in debug mode
            if show_mask or debug_mode:
                cv2.imshow("Green Mask", mask)
            elif cv2.getWindowProperty("Green Mask", cv2.WND_PROP_VISIBLE) > 0:
                cv2.destroyWindow("Green Mask")           
            # Debug mode visualization
            if debug_mode:
                debug_frame = frame.copy()             
                # Get all detected corners before quad selection for debugging
                all_corners = detect_corners_in_mask(mask, detection_config)           
                # Draw all detected corners
                for corner in all_corners:
                    cv2.circle(debug_frame, corner, 3, (0, 255, 255), -1)               
                # Label the corners with index numbers for easier identification
                for i, corner in enumerate(all_corners):
                    cv2.putText(debug_frame, str(i), 
                            (corner[0] + 5, corner[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)                        
                # Draw the current detection parameters on the debug view
                param_text = [
                    f"Hue: {detection_config.get('hue_min', 40)}-{detection_config.get('hue_max', 80)}",
                    f"Sat Min: {detection_config.get('sat_min', 50)}",
                    f"Val Min: {detection_config.get('val_min', 50)}",
                    f"Quality: {detection_config.get('qualityLevel', 0.01):.2f}",
                    f"Square Tolerance: {detection_config.get('square_tolerance', 0.3):.2f}",
                    f"Total corners: {len(all_corners)}"
                ]            
                for i, text in enumerate(param_text):
                    cv2.putText(debug_frame, text, (10, 30 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)         
                # Show the debug view
                cv2.imshow("Debug View", debug_frame)
            else:
                # Close debug windows if they exist
                if cv2.getWindowProperty("Debug View", cv2.WND_PROP_VISIBLE) > 0:
                    cv2.destroyWindow("Debug View")
            
            if auto_green_mode:
                cv2.putText(display_frame, "AUTO GREEN MODE", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Update corner tracker with new quad
            if quad is not None:
                corner_tracker.update(quad)         
            # Get smoothed corners for display
            smoothed_quad = corner_tracker.get_smoothed_corners()
            
            if smoothed_quad is not None:
                # Draw points and quadrilateral lines using smoothed corners
                for point in smoothed_quad:
                    cv2.circle(display_frame, tuple(map(int, point)), 5, (0, 255, 0), -1)            
                # Order the points and draw the quadrilateral
                pts = order_points(smoothed_quad)
                for i in range(4):
                    pt1 = tuple(map(int, pts[i]))
                    pt2 = tuple(map(int, pts[(i+1) % 4]))
                    cv2.line(display_frame, pt1, pt2, (0, 0, 255), 2)          
                # Calculate the center of the quadrilateral
                center_x = int(sum(p[0] for p in smoothed_quad) / 4)
                center_y = int(sum(p[1] for p in smoothed_quad) / 4)
                square_center = (center_x, center_y)     
                # Update trajectory
                trajectory.append(square_center)
                if len(trajectory) > max_trajectory_length:
                    trajectory.pop(0)       
                # Draw a crosshair at the center
                cv2.drawMarker(display_frame, square_center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)      
                # Display "Square Detected" text
                if display_config["show_detection_info"]:
                    cv2.putText(display_frame, "Square Detected!", 
                               (square_center[0] - 60, square_center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)      
                # Optionally show the perspective-corrected view
                if show_perspective and perspective_view is not None:
                    cv2.imshow("Perspective View", perspective_view)
            else:
                # No quadrilateral detected
                if display_config["show_detection_info"]:
                    cv2.putText(display_frame, "No Green Corners Detected", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw target trajectory
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
            
            instructions = "Press 'q' to quit | 'h' for panel | 'p' for perspective | 'm' for mask | 'd' for debug"
            cv2.putText(display_frame, instructions, 
                       (10, display_frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show the display frame
            cv2.imshow("Camera Feed", display_frame)         
            # Use a shorter wait time to improve key responsiveness
            key = cv2.waitKey(10) & 0xFF    
            # Update control panel with key press
            control_panel.update(key)    
            # Check for key presses
            if key == ord('q'):
                print("User requested exit (q key pressed)")
                running = False
                break
            elif key == ord('p'):
                show_perspective = not show_perspective
                if not show_perspective and cv2.getWindowProperty("Perspective View", cv2.WND_PROP_VISIBLE) > 0:
                    cv2.destroyWindow("Perspective View")
            elif key == ord('m'):
                show_mask = not show_mask
                if not show_mask and cv2.getWindowProperty("Green Mask", cv2.WND_PROP_VISIBLE) > 0:
                    cv2.destroyWindow("Green Mask")
            elif key == ord('a'):
                auto_green_mode = not auto_green_mode
                print(f"Auto green detection mode {'enabled' if auto_green_mode else 'disabled'}")
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
                if not debug_mode:
                    if cv2.getWindowProperty("Debug View", cv2.WND_PROP_VISIBLE) > 0:
                        cv2.destroyWindow("Debug View")
                    if not show_mask and cv2.getWindowProperty("Green Mask", cv2.WND_PROP_VISIBLE) > 0:
                        cv2.destroyWindow("Green Mask")
                
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Exiting...")
            running = False
            break
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full stack trace for debugging
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
