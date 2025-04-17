# OpenCV Camera Project: Green Target Tracking

This project provides a robust solution for tracking green markers in a camera feed using OpenCV. It's designed to detect and track a quadrilateral formed by four green markers, providing real-time position data and visualization.

## Features

- **Green Target Detection**: Detects green markers using HSV color filtering
- **Quadrilateral Tracking**: Identifies and tracks four green markers forming a quadrilateral
- **Perspective Correction**: Shows a perspective-corrected view of the detected quadrilateral
- **Interactive Controls**: Adjustable parameters via an on-screen control panel
- **Data Recording**: Records the center point positions of the detected quadrilateral
- **Performance Monitoring**: Real-time FPS and performance statistics

## Requirements

- Python 3.6+
- OpenCV 4.x
- NumPy
- Tkinter (for camera selection GUI)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/opencv-camera-project.git
   cd opencv-camera-project
   ```

2. Install the required dependencies:
   ```bash
   pip install opencv-python numpy
   ```

## Usage

Run the main camera test script:
```bash
python srcs/components/camera/camera_test.py
```

### Key Controls

- `q`: Quit the application
- `h`: Hide/show control panel
- `r`: Reset detection parameters to defaults
- `p`: Toggle perspective view
- `m`: Toggle mask view
- `d`: Toggle debug mode (helps with parameter tuning)
- `a`: Toggle auto green detection mode
- `v`: Start/stop recording square center points

## Tips for Green Marker Detection

- Use bright green markers (post-it notes work well)
- Ensure good lighting conditions
- Adjust Hue Min/Max to match your specific shade of green
- Increase Sat Min if detecting too many non-green objects
- Use debug mode ('d' key) to see what's being detected

## Project Structure

```
opencv-camera-project/
├── srcs/
│   ├── components/
│   │   ├── camera/
│   │   │   ├── camera_handler.py     # Camera initialization and frame capture
│   │   │   ├── camera_selector.py    # GUI for selecting available cameras
│   │   │   └── camera_test.py        # Main application entry point
│   │   └── ui/
│   │       └── control_panel.py      # Interactive parameter adjustment interface
│   └── utils/
│       ├── config.py                 # Configuration management and persistence
│       ├── data_recorder.py          # Records tracking data to CSV files
│       ├── green_corner_detection.py # Core detection and tracking algorithms
│       ├── image_processing.py       # Image manipulation utilities
│       ├── performance_monitor.py    # FPS and performance tracking
│       └── target_detection.py       # Alternative detection methods
└── output/                           # Default directory for recorded data
```

## Key Modules

### Camera Components
- **camera_handler.py**: Manages camera initialization, configuration, and frame capture
- **camera_selector.py**: Provides a GUI for detecting and selecting available cameras
- **camera_test.py**: Main application that integrates all components

### UI Components
- **control_panel.py**: Interactive interface for adjusting detection parameters in real-time

### Utilities
- **config.py**: Manages application configuration and settings persistence
- **data_recorder.py**: Records tracking data to CSV files with timestamps
- **green_corner_detection.py**: Core algorithms for detecting and tracking green markers
- **image_processing.py**: Utilities for image manipulation and transformation
- **performance_monitor.py**: Tracks and reports on application performance metrics
- **target_detection.py**: Alternative detection methods for different tracking scenarios

## Configuration

The application uses a configuration system that allows for adjusting various parameters:

- **Camera settings**: Resolution, FPS
- **Detection settings**: HSV ranges, corner detection parameters
- **Display settings**: Scale factor, information display options

Parameters can be adjusted in real-time using the control panel.

## Data Recording

When recording is enabled (press 'v'), the application saves the center point coordinates of the detected quadrilateral to a CSV file in the output directory. Each recording creates a new file with a timestamp.

## License

???

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


# Running the Camera Project on a Raspberry Pi
The camera project should work on a Raspberry Pi with some adjustments for performance and hardware compatibility. Here are my recommendations:

## Hardware Requirements
1. Raspberry Pi Model:

- Raspberry Pi 4 (2GB+ RAM) recommended for best performance
- Raspberry Pi 3B+ would work with reduced resolution/framerate
2. Camera Options:

- USB webcam (most compatible with existing code)
- Raspberry Pi Camera Module (requires code modification)

3. Display:
- HDMI monitor for GUI interface
- Headless operation possible with modifications

## Setup Instructions
1. Install Dependencies:

```
sudo apt update
sudo apt upgrade
sudo apt install python3-pip python3-opencv
pip3 install numpy
```
2. Performance Optimizations:
- Add these settings to your configuration file:
```
"detection": {
  "processing_scale": 0.5,
  "enable_frame_skip": true
},
"display": {
  "scale_factor": 0.75,
  "show_control_panel": false
}
```
3. Pi Camera Module Integration (if not using USB webcam):
- Modify camera_handler.py to add support for the Pi Camera:
```
# Add to imports
try:
    import picamera
    from picamera.array import PiRGBArray
    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False

# Add to CameraHandler class
def __init__(self, camera_index=0, resolution=(640, 480), fps=30, use_pi_camera=False):
    self.use_pi_camera = use_pi_camera and PI_CAMERA_AVAILABLE
    
    if self.use_pi_camera:
        self.camera = picamera.PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = fps
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        time.sleep(0.1)  # Allow camera to warm up
    else:
        # Existing USB camera code
```
4. Auto-start on Boot (optional):
- Create a systemd service:
```
sudo nano /etc/systemd/system/camera-tracker.service
```
Add:
```
[Unit]
Description=Camera Tracking Application
After=multi-user.target

[Service]
User=pi
WorkingDirectory=/home/pi/opencv-camera-project
ExecStart=/usr/bin/python3 /home/pi/opencv-camera-project/srcs/components/camera/camera_test.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
Enable:
```
sudo systemctl enable camera-tracker.service
```

## Troubleshooting Pi-Specific Issues
1. Performance Issues:

- Lower the resolution to 320x240
- Reduce processing by setting processing_scale to 0.3-0.4
- Disable debug mode and control panel
- Consider using the Pi Camera with direct capture

2. GPIO Integration (optional):

- Add LED indicators for successful detection
 Use buttons for control instead of keyboard

3. Thermal Management:

- Monitor temperature: vcgencmd measure_temp
- Add a fan if running for extended periods
- Add code to reduce processing when temperature is high:
```
def check_temperature():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000.0
        return temp
    except:
        return 0
        
# In main loop
if check_temperature() > 75:  # CPU getting hot
    processing_scale = 0.3  # Reduce processing
```
