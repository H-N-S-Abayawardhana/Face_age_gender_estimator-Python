import cv2
import time

def capture_image():
    """Capture an image from the webcam and save it."""
    # Initialize webcam
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("Error: Could not open webcam")
        return None
    
    # Give the camera a moment to initialize
    time.sleep(1)
    
    # Capture a frame
    ret, frame = cam.read()
    
    if not ret:
        print("Error: Could not capture frame")
        cam.release()
        return None
    
    # Save the captured image
    img_path = "captured.jpg"
    cv2.imwrite(img_path, frame)
    
    # Display the captured image
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(2000)  # Display for 2 seconds
    cv2.destroyAllWindows()
    
    # Release the webcam
    cam.release()
    
    print(f"Image captured and saved to {img_path}")
    return img_path

if __name__ == "__main__":
    capture_image()