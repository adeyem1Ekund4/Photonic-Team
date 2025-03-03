# srcs/utils/performance_monitor.py
import time
import numpy as np
import logging
from typing import Dict

class CameraPerformanceMonitor:
    def __init__(self, log_file: str = 'camera_performance.log'):
        """
        Monitor camera performance and log diagnostics
        
        Args:
            log_file (str): Path to log file
        """
        self.frame_times = []
        self.log_file = log_file
        self.start_time = time.time()
        
        # Configure logging
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s: %(message)s')

    def update_frame_time(self, frame_time: float):
        """
        Record frame processing time
        
        Args:
            frame_time (float): Time taken to process a frame
        """
        self.frame_times.append(frame_time)
        
        # Keep last 100 frame times
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.frame_times:
            return {}
        
        metrics = {
            'avg_fps': len(self.frame_times) / (time.time() - self.start_time),
            'avg_frame_time': np.mean(self.frame_times),
            'max_frame_time': np.max(self.frame_times),
            'min_frame_time': np.min(self.frame_times)
        }
        
        # Log performance metrics periodically
        self._log_performance(metrics)
        
        return metrics

    def _log_performance(self, metrics: Dict[str, float]):
        """
        Log performance metrics
        
        Args:
            metrics (Dict[str, float]): Performance metrics to log
        """
        log_message = " | ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
        logging.info(f"Performance Metrics: {log_message}")
