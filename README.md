# Rock Paper Scissors Classifier Web Application

This is a web application that uses a trained TensorFlow model to classify hand gestures as rock, paper, or scissors using your computer's camera.

## Features

- Real-time camera capture
- Instant classification of rock, paper, scissors gestures
- Confidence score display
- Modern, responsive UI using Tailwind CSS

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Web browser with camera access

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure you have the `rock_paper_scissors_model.h5` file in the project directory
2. Start the Flask server:
   ```bash
   python app.py
   ```
3. Open your web browser and navigate to `http://localhost:5000`
4. Click the "Start Camera" button and allow camera access when prompted
5. Position your hand in front of the camera and click "Capture" to get the prediction

## How to Use

1. Click "Start Camera" to initialize your webcam
2. Position your hand in front of the camera making a rock, paper, or scissors gesture
3. Click "Capture" to take a snapshot and get the prediction
4. The result will show the predicted gesture and the confidence level

## Note

Make sure you have good lighting and a clear view of your hand for best results. The model works best when the hand is centered in the frame and clearly visible. 