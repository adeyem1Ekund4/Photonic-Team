# opencv-camera-project/srcs/utils/image_processing.py
import cv2
import numpy as np

def apply_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    # Brightness adjustment
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    # Contrast adjustment
    if contrast != 0:
        alpha_c = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        gamma_c = 127 * (1 - alpha_c)
        
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    
    # Ensure pixel values are within valid range
    return np.clip(image, 0, 255).astype(np.uint8)

def sharpen_image(image, kernel_size=3, sigma=1.0, amount=1.0):

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def auto_adjust_levels(image):

    # For color images, apply to each channel
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for i in range(3):
            result[:,:,i] = cv2.equalizeHist(image[:,:,i])
        return result
    # For grayscale images
    else:
        return cv2.equalizeHist(image)
    
def resize_frame(frame, scale=0.5):

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
