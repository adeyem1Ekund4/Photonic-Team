# opencv-camera-project/srcs/components/camera/camera_selector.py
import cv2
import tkinter as tk
from tkinter import ttk
import threading
from typing import List, Tuple, Optional

class CameraSelector:
    def __init__(self):
        self.available_cameras = self._find_cameras()
        
    def _find_cameras(self) -> List[Tuple[int, str]]:
        available = []        
        # Check for cameras (typically indices 0-3)
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a test frame
                ret, _ = cap.read()
                if ret:
                    # Get camera name based on index
                    name = "Integrated Camera" if i == 0 else f"USB Camera ({i})"                    
                    # Try to get backend name for more info
                    try:
                        backend = cap.getBackendName()
                        if backend:
                            name += f" - {backend}"
                    except:
                        pass                 
                    # Get resolution for display
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if width > 0 and height > 0:
                        name += f" - {width}x{height}"                    
                    # Add to available cameras list
                    available.append((i, name))            
            cap.release()            
        return available
    
    def select_camera(self) -> Optional[int]:
        if not self.available_cameras:
            print("No cameras detected!")
            return None            
        # If only one camera is available, return it directly
        if len(self.available_cameras) == 1:
            print(f"Only one camera available: {self.available_cameras[0][1]}")
            return self.available_cameras[0][0]        
        # Create a simple GUI for camera selection
        selected_camera = None
        selection_complete = threading.Event()
        
        def on_select():
            nonlocal selected_camera
            selection = camera_combo.current()
            if selection >= 0:
                selected_camera = self.available_cameras[selection][0]
            selection_complete.set()
            root.destroy()
            
        def on_cancel():
            selection_complete.set()
            root.destroy()
        
        # Create the GUI
        root = tk.Tk()
        root.title("Select Camera Source")
        root.geometry("500x250")  # Adjusted size        
        # Center the window on screen
        window_width = 500
        window_height = 250
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')     
        # Main frame with padding
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)   
        # Title label
        title_label = ttk.Label(main_frame, text="Select a camera source:", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))     
        # Dropdown for camera selection
        camera_names = [cam[1] for cam in self.available_cameras]   
        # Label for dropdown
        dropdown_label = ttk.Label(main_frame, text="Available Cameras:", font=("Arial", 12))
        dropdown_label.pack(anchor=tk.W, pady=(5, 10))     
        # Combobox (dropdown)
        camera_combo = ttk.Combobox(main_frame, values=camera_names, width=50, state="readonly")
        camera_combo.current(0)  # Select first camera by default
        camera_combo.pack(fill=tk.X, pady=10)       
        # Description text
        desc_text = "Note: Your UVC webcam should light up orange when connected."
        desc_label = ttk.Label(main_frame, text=desc_text, font=("Arial", 10, "italic"))
        desc_label.pack(pady=(5, 20))      
        # Button frame at the bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))    
        # Style for larger buttons
        style = ttk.Style()
        style.configure('Large.TButton', font=('Arial', 12))       
        # Add prominent buttons
        cancel_button = ttk.Button(button_frame, text="Cancel", command=on_cancel, style='Large.TButton')
        cancel_button.pack(side=tk.LEFT, padx=(0, 10), pady=10, ipadx=10, ipady=5)     
        select_button = ttk.Button(button_frame, text="Start Camera", command=on_select, style='Large.TButton')
        select_button.pack(side=tk.RIGHT, padx=(10, 0), pady=10, ipadx=10, ipady=5)     
        # Make the "Start Camera" button the default action when Enter is pressed
        root.bind('<Return>', lambda event: on_select())
        select_button.focus()     
        # Run the GUI
        root.mainloop()      
        # Wait for selection with timeout
        selection_complete.wait(timeout=30)
        return selected_camera
