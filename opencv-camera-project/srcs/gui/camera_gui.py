import tkinter as tk
import sys
import os

# Add the parent directory of 'srcs' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from components.camera.camera_handler import CameraHandler
from utils.image_processing import apply_grayscale
import cv2

class CameraGUI:
    def __init__(self, master):
        self.master = master
        master.title("Camera Selection")

        # Create a label to display the camera feed
        self.camera_label = tk.Label(master)
        self.camera_label.pack()

        # Create a dropdown menu to select the camera
        self.camera_var = tk.StringVar(master)
        self.camera_var.set("Integrated Camera")
        self.camera_options = ["Integrated Camera", "USB Camera"]
        self.camera_dropdown = tk.OptionMenu(master, self.camera_var, *self.camera_options)
        self.camera_dropdown.pack()

        # Create a button to start the camera
        self.start_button = tk.Button(master, text="Start Camera", command=self.start_camera)
        self.start_button.pack()

        self.camera = None

    def start_camera(self):
        camera_index = 0 if self.camera_var.get() == "Integrated Camera" else 1
        self.camera = CameraHandler(camera_index=camera_index)

        self.update_camera_feed()

    def update_camera_feed(self):
        if self.camera:
            ret, frame = self.camera.get_frame()
            if ret:
                gray_frame = apply_grayscale(frame)
                photo = tk.PhotoImage(data=cv2.imencode('.png', gray_frame)[1].tobytes())
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo
                self.camera_label.after(10, self.update_camera_feed)
            else:
                self.camera.release()
                self.camera = None
                self.camera_label.configure(image=None)

root = tk.Tk()
camera_gui = CameraGUI(root)
root.mainloop()
