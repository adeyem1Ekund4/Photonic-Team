import cv2
import time

def test_windows_camera():
    """
    Simple test script to verify camera access on Windows.
    """
    print("Testing Windows camera access...")
    
    # Try both DirectShow and default
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (None, "Default")
    ]

    for backend, name in backends:
        print(f"\nTrying {name} backend...")
        try:
            if backend:
                cap = cv2.VideoCapture(0, backend)
            else:
                cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print(f"Failed to open camera with {name}")
                continue

            # Try to read 5 frames
            for i in range(5):
                ret, frame = cap.read()
                if ret:
                    print(f"Successfully read frame {i+1}")
                    # Display frame
                    cv2.imshow(f"Test Frame - {name}", frame)
                    cv2.waitKey(500)  # Wait for 500ms
                else:
                    print(f"Failed to read frame {i+1}")

            cap.release()
            cv2.destroyAllWindows()
            print(f"Test with {name} completed")
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            if 'cap' in locals():
                cap.release()

if __name__ == "__main__":
    test_windows_camera()
