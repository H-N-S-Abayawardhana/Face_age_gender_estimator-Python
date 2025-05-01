# Face Age & Gender Estimator

This project uses computer vision and deep learning to detect faces in webcam images and estimate age and gender. It provides both a simple Streamlit web interface and an advanced real-time analysis option.

## Features

- Capture images from your webcam
- Detect faces in the images
- Estimate age and gender of detected faces
- Log predictions for future reference
- Optional real-time face analysis

## Installation

1. Clone this repository or download the files

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic App

Run the basic Streamlit app:
```bash
streamlit run app.py
```

This will open a web interface in your browser where you can:
- Capture an image from your webcam
- Analyze the image for age and gender
- View past predictions

### Advanced App

Run the advanced app with real-time analysis:
```bash
streamlit run advanced_app.py
```

In addition to the basic features, this version allows:
- Real-time face analysis from webcam stream
- More detailed statistics about predictions
- Multiple modes accessible through tabs

## Project Files

- `test_deepface.py` - Test script to verify DeepFace is working correctly
- `capture_webcam.py` - Simple script to capture an image from webcam
- `webcam_analysis.py` - Script to capture and analyze an image in one step
- `app.py` - Basic Streamlit web application
- `advanced_app.py` - Advanced Streamlit application with real-time analysis
- `requirements.txt` - List of required packages
- `prediction_log.csv` - Generated log file of predictions (created when using the app)

## Troubleshooting

### Webcam Access Issues

If you're having trouble accessing your webcam:
- Make sure no other application is using the webcam
- Check your browser permissions for webcam access
- Restart your computer if the webcam was recently in use

### Dependency Issues

If you encounter errors related to missing dependencies:
- Make sure you've installed all required packages: `pip install -r requirements.txt`
- For the advanced app, ensure you have streamlit-webrtc: `pip install streamlit-webrtc`

### Face Detection Issues

If faces aren't being detected:
- Ensure there is sufficient lighting
- Face the camera directly
- Make sure your face is clearly visible and not obscured

## Technical Details

This project uses:
- DeepFace for face analysis
- OpenCV for image capture and processing
- Streamlit for the web interface
- Pandas for data handling
- Streamlit WebRTC for real-time video analysis (advanced app only)