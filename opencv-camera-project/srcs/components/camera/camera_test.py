# opencv-camera-project/srcs/components/camera/camera_test.py

import cv2
import time
import numpy as np
import sys
import os
import serial
from serial.tools import list_ports  # for auto-detecting Maestro port

# Add the project source directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from components.camera.camera_handler import CameraHandler
from components.camera.camera_selector import CameraSelector
from utils.green_corner_detection import (
    detect_green_corners, order_points, CornerTracker,
    detect_corners_in_mask, OPTIMAL_GREEN_PARAMS
)
from utils.image_processing import apply_grayscale, resize_frame
from utils.config import ConfigManager
from utils.performance_monitor import PerformanceMonitor
from components.ui.control_panel import ControlPanel
from utils.data_recorder import DataRecorder

# ─────────── SERVO CALIBRATION & CONTROL ───────────

BAUD_RATE = 9600\

ser = None
'''
def find_maestro_port():
    """
    Scan all serial ports and return the first one whose
    description or device name looks like a Pololu Maestro.
    Falls back to the first available port if nothing obvious is found.
    """
    ports = list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        dev  = (p.device or "").lower()
        # match common Pololu identifiers or generic USB-serial
        if "pololu" in desc or "maestro" in desc \
           or dev.startswith(("com", "/dev/ttyacm", "/dev/ttyusb")):
           if "servo controller" in desc and "ttl" not in desc:
            return p.device
    # fallback to first port
    return ports[0].device if ports else None
'''
#hard coded to do COM6
def find_maestro_port() -> str:
    """
    Return the hard-coded serial port for the Pololu Maestro.
    """
    return "COM6"

def init_servo():
    """user defined connection to the Maestro, and let it reset."""
    global ser
    port = find_maestro_port()
    if port is None:
        raise RuntimeError("No serial ports detected. Is your Maestro plugged in?")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(0.2)
        print(f"Connected to Maestro on port {port}")
    except serial.SerialException as e:
        raise RuntimeError(f"Failed to open serial port {port}: {e}")

def set_target(ch, micro_ms):
    """
    Send a Set Target (0x84) to channel ch
    with pulse width micro_ms milliseconds.
    """
    if ser is None:
        return
    tgt = int(micro_ms * 1000 * 4)  # convert ms → quarter-µs units
    ser.write(bytes([0x84, ch, tgt & 0x7F, (tgt >> 7) & 0x7F]))

def track_to_servos(cx, cy, frame_w, frame_h):
    """
    Map pixel coords (cx,cy) to servo angles and send commands:
      • ch 0: horizontal, 270° range, mid=135°
      • ch 1: vertical,   180° range, mid=90°
    Scalars from Calibration sliders adjust edge sensitivity.
    """
    global theta_scalar, phi_scalar
    # read scalars from trackbars
    theta_scalar = cv2.getTrackbarPos("Theta×100", "Calibration") / 100.0
    phi_scalar   = cv2.getTrackbarPos("Phi×100",   "Calibration") / 100.0

    # normalize coords to [-1..+1]
    x_norm = (cx - frame_w/2) / (frame_w/2)
    y_norm = (cy - frame_h/2) / (frame_h/2)

    # HORIZONTAL (ch 0): invert mapping for reversed direction
    h_mid  = 135
    h_half = 135 * theta_scalar
    h_angle = h_mid - x_norm * h_half
    h_angle = max(0.0, min(270.0, h_angle))
    pulse_h = 0.5 + (h_angle / 270.0) * 2.0

    # VERTICAL (ch 1)
    v_mid  = 90
    v_half = 90 * phi_scalar
    v_angle = v_mid - y_norm * v_half
    v_angle = max(0.0, min(180.0, v_angle))
    pulse_v = 0.5 + (v_angle / 180.0) * 2.0

    set_target(0, pulse_h)
    set_target(1, pulse_v)

# ────────────────────────────────────────────────────────

# Scalars adjust how camera edges map to servo edges.
# 1.0 means edges of FOV → edges of servo travel.
theta_scalar = 0.23  # horizontal (270° servo) #hard coded to 23
phi_scalar   = 0.26  # vertical   (180° servo) #hard coded to 26

def main():
    print("Initializing Camera Tracking Application...")

    # Load configs
    config_manager   = ConfigManager()
    camera_config    = config_manager.get_camera_config()
    detection_config = config_manager.get_detection_config()
    display_config   = config_manager.get_display_config()

    # Performance monitor
    performance_monitor = PerformanceMonitor()

    # Init servo & Calibration UI
    init_servo()
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Theta×100", "Calibration", int(theta_scalar*100), 200, lambda x: None)
    cv2.createTrackbar("Phi×100",   "Calibration", int(phi_scalar*100),   200, lambda x: None)

    # Camera selection
    camera_selector = CameraSelector()
    selected_camera = camera_selector.select_camera()
    if selected_camera is None:
        print("No camera selected. Exiting.")
        return

    # Camera init
    try:
        camera = CameraHandler(
            camera_index=selected_camera,
            resolution=tuple(camera_config["resolution"]),
            fps=camera_config["fps"]
        )
    except Exception as e:
        print(f"Error initializing camera: {e}\nExiting.")
        return

    # UI panel, tracker, recorder
    control_panel   = ControlPanel(config_manager)
    corner_tracker  = CornerTracker(history_length=detection_config.get("history_length", 5))
    data_recorder   = DataRecorder(
        output_dir      = config_manager.get_save_config().get("output_directory", "output"),
        filename_prefix = config_manager.get_save_config().get("filename_prefix", "xy_greentarget_"),
        interval_ms     = 250
    )

    # Flags & state
    debug_mode       = False
    auto_green_mode  = False
    show_perspective = display_config.get("show_perspective", False)
    show_mask        = False
    running          = True
    trajectory       = []
    max_trajectory_length = detection_config.get("history_length", 5)
    frame_skip_threshold   = 0.05
    last_processing_time   = 0.01

    # On-screen help
    print("Camera initialized. Press 'q' to quit.")
    print("Press 'h' to hide/show control panel.")
    print("Press 'r' to reset detection parameters to defaults.")
    print("Press 'p' to toggle perspective view.")
    print("Press 'm' to toggle mask view.")
    print("Press 'd' to toggle debug mode (helps with parameter tuning).")
    print("Press 'a' to toggle auto green detection mode.")
    print("Press 'v' to start/stop recording square center points")
    print("NOTE: Make sure the camera window is in focus when pressing keys")
    print("\nTIPS FOR GREEN MARKER DETECTION:")
    print("1. Use bright green markers (post-it notes work well)")
    print("2. Ensure good lighting conditions")
    print("3. Adjust Hue Min/Max to match your specific shade of green")
    print("4. Increase Sat Min if detecting too many non-green objects")
    print("5. Use debug mode ('d' key) to see what's being detected")

    while running:
        try:
            t0 = time.time()

            # Adaptive frame-skip
            if (last_processing_time > frame_skip_threshold
                and config_manager.get_detection_config().get("enable_frame_skip", False)):
                processing_scale = 0.5
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                processing_scale = detection_config.get("processing_scale", 0.75)

            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Resize
            if display_config["scale_factor"] != 1.0:
                frame = resize_frame(frame, scale=display_config["scale_factor"])
            display_frame    = frame.copy()
            processing_frame = (resize_frame(frame, scale=processing_scale)
                                if processing_scale != 1.0 else frame.copy())

            # Refresh detection config
            detection_config = config_manager.get_detection_config()

            # Detect corners
            try:
                quad, perspective_view, mask = detect_green_corners(processing_frame, detection_config)
                if processing_scale != 1.0 and quad is not None:
                    ratio = 1.0 / processing_scale
                    quad = [(int(x*ratio), int(y*ratio)) for x, y in quad]
            except Exception as e:
                print(f"Detection error: {e}")
                continue

            # Show/hide mask
            if show_mask or debug_mode:
                cv2.imshow("Green Mask", mask)
            else:
                if cv2.getWindowProperty("Green Mask", cv2.WND_PROP_VISIBLE) > 0:
                    cv2.destroyWindow("Green Mask")

            # Debug view
            if debug_mode:
                debug_frame = frame.copy()
                all_corners = detect_corners_in_mask(mask, detection_config)
                for i, c in enumerate(all_corners):
                    cv2.circle(debug_frame, c, 3, (0,255,255), -1)
                    cv2.putText(debug_frame, str(i), (c[0]+5,c[1]+5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1)
                # parameters overlay
                param_text = [
                    f"Hue: {detection_config.get('hue_min')}–{detection_config.get('hue_max')}",
                    f"SatMin: {detection_config.get('sat_min')}",
                    f"ValMin: {detection_config.get('val_min')}",
                    f"Quality: {detection_config.get('qualityLevel'):.2f}",
                    f"Tolerance: {detection_config.get('square_tolerance'):.2f}",
                    f"Total corners: {len(all_corners)}",
                    f"Proc time: {last_processing_time*1000:.1f} ms"
                ]
                for i, txt in enumerate(param_text):
                    cv2.putText(debug_frame, txt, (10, 30 + i*25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
                cv2.imshow("Debug View", debug_frame)
            else:
                if cv2.getWindowProperty("Debug View", cv2.WND_PROP_VISIBLE) > 0:
                    cv2.destroyWindow("Debug View")

            # Auto-green indicator
            if auto_green_mode:
                cv2.putText(display_frame, "AUTO GREEN MODE", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

            # Corner smoothing + drawing
            if quad is not None:
                corner_tracker.update(quad)
            smoothed_quad = corner_tracker.get_smoothed_corners()

            if smoothed_quad is not None:
                for p in smoothed_quad:
                    cv2.circle(display_frame, tuple(map(int,p)), 5, (0,255,0), -1)
                pts = order_points(smoothed_quad)
                for i in range(4):
                    cv2.line(display_frame,
                             tuple(map(int,pts[i])),
                             tuple(map(int,pts[(i+1)%4])),
                             (0,0,255),2)
                cx = int(sum(p[0] for p in smoothed_quad)/4)
                cy = int(sum(p[1] for p in smoothed_quad)/4)
                square_center = (cx, cy)
                trajectory.append(square_center)
                if len(trajectory) > max_trajectory_length:
                    trajectory.pop(0)
                cv2.drawMarker(display_frame, square_center, (0,0,255),
                               cv2.MARKER_CROSS, 20,2)
                if display_config["show_detection_info"]:
                    cv2.putText(display_frame, "Square Detected!",
                                (cx-60, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0,255,0),2)
                if show_perspective and perspective_view is not None:
                    cv2.imshow("Perspective View", perspective_view)

                # Servo tracking
                track_to_servos(cx, cy,
                                display_frame.shape[1],
                                display_frame.shape[0])
            else:
                if display_config["show_detection_info"]:
                    cv2.putText(display_frame, "No Green Corners Detected",
                                (10,60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0,0,255),2)

            # Trajectory lines
            for i in range(1, len(trajectory)):
                intensity = int(255 * (i / len(trajectory)))
                cv2.line(display_frame, trajectory[i-1], trajectory[i],
                         (0,intensity,255-intensity),2)

            # Performance timing
            frame_elapsed = time.time() - t0
            performance_monitor.update(frame_elapsed)
            last_processing_time = frame_elapsed

            # FPS display
            if display_config["show_fps"]:
                fps = performance_monitor.get_fps()
                cv2.putText(display_frame, f"FPS: {fps:.1f}",
                            (10, display_frame.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

            # On-screen instructions
            instr = ("Press 'q' to quit | 'h' panel | 'r' reset | "
                     "'p' perspec | 'm' mask | 'd' debug | "
                     "'a' auto | 'v' rec")
            cv2.putText(display_frame, instr,
                        (10, display_frame.shape[0]-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),2)

            # Data recording
            if smoothed_quad is not None:
                data_recorder.record_point((cx,cy))
                if data_recorder.is_recording:
                    cv2.putText(display_frame, "RECORDING",
                                (display_frame.shape[1]-120,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

            cv2.imshow("Camera Feed", display_frame)

            # Key handling
            key = cv2.waitKey(10) & 0xFF
            control_panel.update(key)
            if key == ord('q'):
                running = False
            elif key == ord('h'):
                control_panel.toggle_visibility()
            elif key == ord('r'):
                config_manager.reset_detection_defaults()
            elif key == ord('p'):
                show_perspective = not show_perspective
                if not show_perspective:
                    cv2.destroyWindow("Perspective View")
            elif key == ord('m'):
                show_mask = not show_mask
            elif key == ord('d'):
                debug_mode = not debug_mode
            elif key == ord('a'):
                auto_green_mode = not auto_green_mode
            elif key == ord('v'):
                if not data_recorder.is_recording:
                    data_recorder.start_recording()
                else:
                    data_recorder.stop_recording()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback; traceback.print_exc()
            time.sleep(1)

    # ─── CLEANUP ───
    print("Cleaning up resources…")
    if data_recorder.is_recording:
        data_recorder.stop_recording()
    camera.release()
    control_panel.close()
    cv2.destroyAllWindows()
    if ser:
        ser.close()

    stats = performance_monitor.get_stats()
    print("\nPerformance Summary:")
    print(f"  Average FPS:            {stats['overall_fps']:.2f}")
    print(f"  Total frames processed: {stats['total_frames']}")
    print(f"  Total runtime:          {stats['total_runtime']:.2f} s")
    print(f"  Avg frame time:         {stats['avg_frame_time']*1000:.2f} ms")
    print("Application terminated successfully")


if __name__ == '__main__':
    main()
