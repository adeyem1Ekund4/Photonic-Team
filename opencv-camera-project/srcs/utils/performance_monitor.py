# opencv-camera-project/srcs/utils/performance_monitor.py
import time
import numpy as np
import logging
from typing import Dict

class PerformanceMonitor:
    def __init__(self, history_length=100):
        """
        Monitor performance metrics like FPS and frame processing time.
        
        Args:
            history_length (int): Number of frames to keep in history for calculations
        """
        self.frame_times = []
        self.history_length = history_length
        self.start_time = time.time()
        self.total_frames = 0

    def update(self, frame_time: float):
        """
        Update with a new frame processing time
        
        Args:
            frame_time (float): Time taken to process the frame in seconds
        """
        self.frame_times.append(frame_time)
        self.total_frames += 1
        
        # Keep history limited to specified length
        if len(self.frame_times) > self.history_length:
            self.frame_times.pop(0)

    def get_fps(self) -> float:
        """
        Calculate current frames per second based on recent history
        
        Returns:
            float: Current FPS
        """
        if not self.frame_times:
            return 0.0
        
        # Calculate FPS from average frame time
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        if avg_frame_time > 0:
            return 1.0 / avg_frame_time
        return 0.0

    def get_overall_fps(self) -> float:
        """
        Calculate overall average FPS since monitoring started
        
        Returns:
            float: Overall average FPS
        """
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            return self.total_frames / elapsed_time
        return 0.0

    def get_stats(self) -> Dict[str, float]:
        """
        Get comprehensive performance statistics
        
        Returns:
            Dict: Dictionary of performance metrics
        """
        if not self.frame_times:
            return {
                "current_fps": 0.0,
                "overall_fps": 0.0,
                "avg_frame_time": 0.0,
                "min_frame_time": 0.0,
                "max_frame_time": 0.0,
                "total_runtime": time.time() - self.start_time,
                "total_frames": self.total_frames
            }
        
        return {
            "current_fps": self.get_fps(),
            "overall_fps": self.get_overall_fps(),
            "avg_frame_time": sum(self.frame_times) / len(self.frame_times),
            "min_frame_time": min(self.frame_times),
            "max_frame_time": max(self.frame_times),
            "total_runtime": time.time() - self.start_time,
            "total_frames": self.total_frames
        }

    def log_stats(self, interval=60):
        """
        Log statistics at specified intervals
        
        Args:
            interval (int): Logging interval in seconds
        """
        elapsed = time.time() - self.start_time
        if elapsed > 0 and int(elapsed) % interval == 0:
            stats = self.get_stats()
            log_message = (
                f"Performance Stats | "
                f"Current FPS: {stats['current_fps']:.2f} | "
                f"Overall FPS: {stats['overall_fps']:.2f} | "
                f"Avg Frame Time: {stats['avg_frame_time']*1000:.2f}ms | "
                f"Total Frames: {stats['total_frames']}"
            )
            print(log_message)
