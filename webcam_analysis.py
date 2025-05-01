import cv2
import time
from deepface import DeepFace

def capture_and_analyze():
    """Capture an image from webcam and analyze age and gender."""
    # Initialize webcam
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Give the camera a moment to initialize
    time.sleep(1)
    
    # Capture a frame
    ret, frame = cam.read()
    
    if not ret:
        print("Error: Could not capture frame")
        cam.release()
        return
    
    # Save the captured image
    img_path = "captured.jpg"
    cv2.imwrite(img_path, frame)
    
    # Release the webcam
    cam.release()
    
    print(f"Image captured and saved to {img_path}")
    
    # Display the image
    cv2.imshow("Analyzing...", frame)
    cv2.waitKey(1)  # This keeps the window open while analyzing
    
    try:
        # Analyze the image with DeepFace
        print("Analyzing image...")
        result = DeepFace.analyze(img_path=img_path, actions=["age", "gender"])
        
        # Extract and display results
        age = result[0]["age"]
        gender = result[0]["gender"]
        gender_confidence = result[0]["gender_confidence"]
        
        print(f"Age: {age}")
        print(f"Gender: {gender} ({gender_confidence:.2f}% confidence)")
        
        # Draw results on the image
        text = f"Age: {age}, Gender: {gender}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow("Analysis Result", frame)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_analyze()