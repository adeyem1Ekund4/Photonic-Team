# README.md
# OpenCV Camera Project

## Overview

This project is designed to capture video from a camera, process the frames to detect specific targets, and save the detected target coordinates. It uses OpenCV for image processing and camera handling.

## Features

- Capture video from available cameras.
- Convert frames to grayscale.
- Detect the brightest spot in the frame, assumed to be a target.
- Map detected target coordinates to a different coordinate system.
- Save target coordinates with timestamps to a file.

## Installation

1. Clone the repository.
2. Navigate to the project directory.
3. Install the required packages using:
   ```bash
   pip install -r requirements.txt

## Usage

To run the camera test, execute the following command:

   python srcs/components/camera/camera_test.py

Press 'q' or 'Ctrl/Cmd+C' to quit the application. The detected target coordinates will be saved in a file named target_coordinates.txt

## New Version (2025)

1. Camera Handling
- CameraHandler: Manages camera initialization, frame capture, and camera properties
- CameraSelector: Detects available cameras and provides a GUI for camera selection
2. Target Detection and Tracking
Multiple detection methods:
- Brightness-based detection for finding bright dots
- Contour-based detection for finding circular shapes
- Hybrid approach combining both methods
- TargetTracker: Tracks square targets over time using a Kalman filter for smooth tracking and prediction
3. Configuration Management
- ConfigManager: Handles loading/saving settings from a JSON file with sensible defaults
- Manages camera, detection, display, and output settings
4. UI Components
ControlPanel: Provides sliders and controls to adjust detection parameters in real-time

# Key Features
- Multi-method target detection - Can detect targets using brightness, contour analysis, or a hybrid approach
- Square pattern recognition - Identifies when four dots form a square arrangement
- Predictive tracking - Uses Kalman filtering to predict target position when detection fails
- Real-time parameter adjustment - Interactive control panel for tuning detection parameters
- Performance monitoring - Tracks FPS and processing times
