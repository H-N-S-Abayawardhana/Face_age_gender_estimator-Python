from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# Path to a sample image - replace with your own image path
img_path = "sample.jpg"  # You'll need to add a sample image to your project directory

# Analyze the image
try:
    result = DeepFace.analyze(img_path=img_path, actions=["age", "gender"])
    
    # Display results
    print("Analysis Results:")
    print(f"Age: {result[0]['age']}")
    print(f"Gender: {result[0]['gender']} ({result[0]['gender_confidence']:.2f}% confidence)")
    
    # Display the image with OpenCV
    img = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Age: {result[0]['age']}, Gender: {result[0]['gender']}")
    plt.show()
    
except Exception as e:
    print(f"Error: {e}")