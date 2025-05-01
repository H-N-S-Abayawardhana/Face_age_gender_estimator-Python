import streamlit as st
import cv2
import time
import os
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Face Age & Gender Estimator",
    page_icon="🔍",
    layout="centered"
)

# Define function to capture image from webcam
def capture_from_webcam():
    """Capture image from webcam and return the frame"""
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        st.error("Could not access webcam. Please make sure it's connected and not in use by another application.")
        return None
    
    # Give camera time to initialize
    time.sleep(1)
    
    # Capture frame
    ret, frame = cam.read()
    cam.release()
    
    if not ret:
        st.error("Failed to capture image from webcam.")
        return None
        
    return frame

# Define function to save prediction to log file
def log_prediction(age, gender):
    """Save prediction to a CSV log file"""
    log_file = "prediction_log.csv"
    
    # Create the file with headers if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("timestamp,age,gender\n")
    
    # Append the new prediction
    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp},{age},{gender}\n")

# App title and introduction
st.title("🖼️ Face Age & Gender Estimator")
st.write("""
This app captures an image from your webcam and uses AI to estimate the age and gender 
of any face detected in the image. Simply click the button below to start!
""")

# Create two columns for layout
col1, col2 = st.columns([3, 2])

# Capture button
if st.button("📸 Capture from Webcam", use_container_width=True):
    with st.spinner("Accessing webcam..."):
        frame = capture_from_webcam()
        
    if frame is not None:
        # Save captured image
        img_path = "captured.jpg"
        cv2.imwrite(img_path, frame)
        
        # Display captured image (convert BGR to RGB for proper display)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Image", use_column_width=True)
        
        # Analyze the image
        with st.spinner("Analyzing image..."):
            try:
                result = DeepFace.analyze(img_path=img_path, actions=["age", "gender"])
                age = result[0]["age"]
                gender = result[0]["gender"]
                gender_confidence = result[0]["gender_confidence"]
                
                # Draw a box around the face if available in result
                if "region" in result[0]:
                    region = result[0]["region"]
                    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add text with results
                    text = f"Age: {age}, Gender: {gender}"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Save and display the annotated image
                    cv2.imwrite("analyzed.jpg", frame)
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Analysis Result", use_column_width=True)
                
                # Display results
                st.success(f"✅ Analysis complete!")
                
                # Create columns for results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Estimated Age", f"{age} years")
                with col2:
                    st.metric("Predicted Gender", f"{gender} ({gender_confidence:.1f}%)")
                
                # Log the prediction
                log_prediction(age, gender)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Make sure there is a clearly visible face in the image.")

# Display the log if it exists
if os.path.exists("prediction_log.csv"):
    st.subheader("📊 Previous Predictions")
    try:
        log_df = pd.read_csv("prediction_log.csv")
        if not log_df.empty:
            st.dataframe(log_df, use_container_width=True)
        else:
            st.info("No predictions logged yet.")
    except Exception:
        st.info("No predictions logged yet.")

# Add some information about the app
with st.expander("ℹ️ About this App"):
    st.write("""
    This application uses:
    - **OpenCV** for webcam capture
    - **DeepFace** for face analysis
    - **Streamlit** for the web interface
    
    The age and gender predictions are based on machine learning models and may not always be 100% accurate.
    """)