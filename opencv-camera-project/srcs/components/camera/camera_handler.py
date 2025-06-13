# opencv-camera-project/srcs/components/camera/camera_handler.py
import cv2

class CameraHandler:
    """
    OOP Camera abstraction for easy management and resource safety.
    Supports context management.
    """
    def __init__(self, camera_index=0, resolution=(640, 480), fps=30):
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.open()

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open camera with index {self.camera_index}")
        self.set_resolution(*self.resolution)
        self.set_fps(self.fps)

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def set_resolution(self, width, height):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

    def set_fps(self, fps):
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def get_properties(self):
        return {
            "resolution": self.resolution,
            "fps": self.fps,
            "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation": self.cap.get(cv2.CAP_PROP_SATURATION),
            "hue": self.cap.get(cv2.CAP_PROP_HUE),
            "gain": self.cap.get(cv2.CAP_PROP_GAIN),
            "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE)
        }

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
