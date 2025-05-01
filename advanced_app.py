import streamlit as st
import cv2
import time
import os
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import numpy as np
import threading
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Set page configuration
st.set_page_config(
    page_title="Advanced Face Age & Gender Estimator",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "captured_frame" not in st.session_state:
    st.session_state.captured_frame = None

# Function to capture image from webcam (standard method)
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

# Function to analyze image
def analyze_image(image, save_log=True):
    """Analyze image for age and gender using DeepFace"""
    # Create a temporary file to save the image
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        cv2.imwrite(tmp_path, image)
    
    try:
        # Analyze the image
        result = DeepFace.analyze(img_path=tmp_path, actions=["age", "gender"], enforce_detection=False)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        if not result:
            return None
            
        # Extract results
        age = result[0]["age"]
        gender = result[0]["gender"]
        gender_confidence = result[0]["gender_confidence"]
        
        # Log the prediction if requested
        if save_log:
            log_prediction(age, gender)
            
        # Return results
        return {
            "age": age,
            "gender": gender,
            "confidence": gender_confidence,
            "region": result[0].get("region", None)
        }
        
    except Exception as e:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e

# Function to log predictions
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

# Define WebRTC video processor for real-time analysis
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_analysis_time = time.time()
        self.analysis_result = None
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Analyze every 30 frames (adjust as needed)
        current_time = time.time()
        if current_time - self.last_analysis_time > 3:  # analyze every 3 seconds
            try:
                # Run analysis in a separate thread to avoid blocking
                self.analysis_result = analyze_image(img, save_log=False)
                self.last_analysis_time = current_time
            except Exception as e:
                pass
                
        # Draw results on frame if available
        if self.analysis_result and "region" in self.analysis_result:
            region = self.analysis_result["region"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add text with results
            age = self.analysis_result["age"]
            gender = self.analysis_result["gender"]
            text = f"Age: {age}, Gender: {gender}"
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame.from_ndarray(img)

# App title and introduction
st.title("🖼️ Advanced Face Age & Gender Estimator")
st.write("""
This app uses AI to estimate age and gender from faces captured with your webcam.
Choose between a single capture or real-time analysis!
""")

# Create tabs for different modes
tab1, tab2, tab3 = st.tabs(["📸 Single Capture", "🎥 Real-time Analysis", "📊 Prediction Log"])

# Single capture tab
with tab1:
    st.subheader("Take a single photo for analysis")
    
    if st.button("📸 Capture from Webcam", key="single_capture", use_container_width=True):
        with st.spinner("Accessing webcam..."):
            frame = capture_from_webcam()
            
        if frame is not None:
            # Save the frame in session state
            st.session_state.captured_frame = frame
            
            # Display captured image
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Image", use_column_width=True)
            
            # Analyze the image
            with st.spinner("Analyzing image..."):
                try:
                    result = analyze_image(frame)
                    if result:
                        st.session_state.analysis_results = result
                        
                        # Display results
                        st.success(f"✅ Analysis complete!")
                        
                        # Create columns for results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Estimated Age", f"{result['age']} years")
                        with col2:
                            st.metric("Predicted Gender", f"{result['gender']} ({result['confidence']:.1f}%)")
                            
                        # Draw box on image if region information is available
                        if "region" in result:
                            annotated_frame = frame.copy()
                            region = result["region"]
                            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # Add text with results
                            text = f"Age: {result['age']}, Gender: {result['gender']}"
                            cv2.putText(annotated_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            # Display annotated image
                            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Analysis Result", use_column_width=True)
                    else:
                        st.warning("No face detected in the image.")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Make sure there is a clearly visible face in the image.")

# Real-time analysis tab
with tab2:
    st.subheader("Real-time webcam analysis")
    st.write("This mode will analyze faces in real-time through your webcam stream.")
    
    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="face-analysis",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )
    
    if webrtc_ctx.video_processor:
        st.info("Real-time analysis is active! Look at the video stream to see results.")
        st.warning("Note: For improved performance, analysis is performed every few seconds.")
        
        # Add a button to capture and save the current analysis
        if st.button("📊 Save Current Analysis", key="save_analysis"):
            if webrtc_ctx.video_processor.analysis_result:
                result = webrtc_ctx.video_processor.analysis_result
                # Log the prediction
                log_prediction(result["age"], result["gender"])
                st.success("Analysis saved to log!")
            else:
                st.warning("No analysis available yet. Please wait for a face to be detected.")

# Prediction log tab
with tab3:
    st.subheader("Previous Predictions")
    
    if os.path.exists("prediction_log.csv"):
        try:
            log_df = pd.read_csv("prediction_log.csv")
            if not log_df.empty:
                # Convert timestamp to datetime if it's not already
                if log_df["timestamp"].dtype == object:
                    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
                
                # Display the dataframe
                st.dataframe(log_df, use_container_width=True)
                
                # Add some basic statistics
                st.subheader("Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Predictions", len(log_df))
                    st.metric("Average Age", f"{log_df['age'].mean():.1f} years")
                
                with col2:
                    gender_counts = log_df["gender"].value_counts()
                    st.metric("Most Common Gender", gender_counts.index[0])
                    st.metric("% of Predictions", f"{(gender_counts.iloc[0]/len(log_df))*100:.1f}%")
                
                # Add a clear log button
                if st.button("🗑️ Clear Log", key="clear_log"):
                    os.remove("prediction_log.csv")
                    st.success("Log cleared!")
                    st.experimental_rerun()
            else:
                st.info("No predictions logged yet.")
        except Exception as e:
            st.error(f"Error loading log: {str(e)}")
    else:
        st.info("No predictions logged yet.")

# Add some information about the app
with st.expander("ℹ️ About this App"):
    st.write("""
    This application uses:
    - **OpenCV** for webcam capture
    - **DeepFace** for face analysis
    - **Streamlit** for the web interface
    - **Streamlit WebRTC** for real-time video streaming
    
    The age and gender predictions are based on machine learning models and may not always be 100% accurate.
    For the real-time analysis mode, you'll need to install additional dependencies:
    ```
    pip install streamlit-webrtc
    ```
    """)