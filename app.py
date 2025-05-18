from flask import Flask, request, jsonify, render_template, Response
import tensorflow as tf
import numpy as np
import cv2
import os
import base64
import time
from PIL import Image
import io
import re

app = Flask(__name__, static_folder='static', template_folder='templates')

# Global variables for model and class names
model = None
class_names = ['rock', 'paper', 'scissors']

# Load the model
def load_model_safely():
    global model
    try:
        # Try to load the .keras format model first (prioritized)
        model_path = 'rock_paper_scissors_model.keras'
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            # Use compile=False to avoid compilation issues when loading
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully from .keras file")
            return True
        
        # Try SavedModel format
        model_dir = 'rock_paper_scissors_model'
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            print(f"Loading model from directory {model_dir}")
            model = tf.keras.models.load_model(model_dir, compile=False)
            print("Model loaded successfully from SavedModel directory")
            return True
            
        print("Model file not found. Please ensure your model is saved as 'rock_paper_scissors_model.keras'")
        return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Process image data
def preprocess_image(image_data):
    try:
        # Remove header from base64 data
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 string to image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert from BGR to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img_resized = cv2.resize(img_array, (150, 150))
        
        # Normalize pixel values
        img_normalized = img_resized / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get image data from request
            data = request.get_json()
            image_data = data.get('image', '')
            
            # Check if model is loaded
            if model is None:
                return jsonify({'error': 'Model not loaded. Please restart the server.'})
            
            # Process the image
            processed_image = preprocess_image(image_data)
            if processed_image is None:
                return jsonify({'error': 'Could not process the image'})
            
            # Make prediction
            prediction = model.predict(processed_image, verbose=0)
            print('Prediction array:', prediction[0])  # Debug: print raw prediction
            predicted_class_idx = np.argmax(prediction[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = float(prediction[0][predicted_class_idx])
            
            # Return prediction result
            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/video_feed')
def video_feed():
    # Video streaming route
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success:
            break
        else:
            # Optional: Process frame for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_frame, (150, 150))
            normalized = resized / 255.0
            img_batch = np.expand_dims(normalized, axis=0)
            
            # Make prediction if model is loaded
            if model is not None:
                prediction = model.predict(img_batch, verbose=0)
                predicted_class_idx = np.argmax(prediction[0])
                predicted_class = class_names[predicted_class_idx]
                confidence = float(prediction[0][predicted_class_idx]) * 100
                
                # Add prediction text to frame
                cv2.putText(frame, f"{predicted_class.upper()}: {confidence:.1f}%", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield the frame in the byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    # Load model before starting the server
    if load_model_safely():
        print("Model loaded successfully. Starting server...")
        app.run(debug=True, host='0.0.0.0')
    else:
        print("Failed to load model. Server not started.")