# opencv-camera-project/srcs/components/camera/camera_handler.py
import cv2

class CameraHandler:
    def __init__(self, camera_index=0, resolution=(640, 480), fps=30):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open camera with index {camera_index}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Set FPS
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Verify settings were applied
        self.actual_resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized with resolution: {self.actual_resolution}, FPS: {self.actual_fps}")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def get_camera_properties(self):
        """Return the current camera properties."""
        return {
            "resolution": self.actual_resolution,
            "fps": self.actual_fps,
            "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation": self.cap.get(cv2.CAP_PROP_SATURATION),
            "hue": self.cap.get(cv2.CAP_PROP_HUE),
            "gain": self.cap.get(cv2.CAP_PROP_GAIN),
            "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE)
        }

    def set_resolution(self, width, height):
        """Set camera resolution and return if it was successful."""
        result_width = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        result_height = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Update actual resolution
        self.actual_resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        
        return result_width and result_height

    def set_fps(self, fps):
        """Set camera FPS and return if it was successful."""
        result = self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Update actual FPS
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        return result

    def release(self):
        self.cap.release()
