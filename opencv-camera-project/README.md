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

Press 'q' to quit the application. The detected target coordinates will be saved in a file named target_coordinates.txt