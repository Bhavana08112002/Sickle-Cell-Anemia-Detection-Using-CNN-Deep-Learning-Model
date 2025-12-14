import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    BatchNormalization
)
from tensorflow.keras.applications import MobileNetV2
import tempfile
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Global variable to store the model
model = None
model_type = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_model2():
    """Create Model 2 (best performing with batch norm and dropout)"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model

def create_model4():
    """Create Model 4 (Transfer learning with MobileNetV2)"""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(256,256,3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False

    model = Sequential([
        base,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model

def load_or_create_model():
    """Load trained model or create new one"""
    global model, model_type
    
    model_path = "model_checkpoint.h5"
    
    # Check if trained model exists
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            model_type = "loaded"
            print(f"✓ Loaded TRAINED model from {model_path}")
            print(f"✓ Model is ready for accurate predictions")
            return True
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Using untrained model instead")
    
    # Create model for inference (untrained)
    print("\n" + "⚠"*30)
    print("⚠  WARNING: NO TRAINED MODEL FOUND!")
    print("⚠  The app is using an UNTRAINED model")
    print("⚠  Predictions will be RANDOM and INACCURATE")
    print("⚠")
    print("⚠  To fix this:")
    print("⚠  1. Run: python Sickle_cell_anemia.py")
    print("⚠  2. This will train and save model_checkpoint.h5")
    print("⚠  3. Restart this Flask app")
    print("⚠"*30 + "\n")
    
    try:
        model = create_model2()
        model_type = "untrained"
        print("⚠ Created new untrained Model 2 for testing only")
        return True
    except Exception as e:
        print(f"Error creating model: {e}")
        return False

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
        
        # Resize to model input size
        img = cv2.resize(img, (256, 256))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        raise Exception(f"Image preprocessing error: {str(e)}")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """Classify uploaded image"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded. Please restart the server.'}), 500
        
        # Check if file is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        # Save temporary file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Preprocess image
            processed_image = preprocess_image(temp_path)
            
            # Make prediction
            prediction = model.predict(processed_image, verbose=0)
            confidence = float(prediction[0][0])
            
            # Determine classification
            # Assuming: confidence > 0.5 = Sickle Cell, else = Normal
            if confidence > 0.5:
                classification = "Sickle Cell"
                class_confidence = confidence
            else:
                classification = "Normal"
                class_confidence = 1 - confidence
            
            # Prepare response
            response = {
                'classification': classification,
                'confidence': class_confidence,
                'model_type': model_type,
                'raw_prediction': confidence
            }
            
            # Cleanup
            os.remove(temp_path)
            
            return jsonify(response), 200
        
        except Exception as e:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return jsonify({'error': f'Classification error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_type': model_type
    }), 200

@app.route('/info', methods=['GET'])
def info():
    """Get application info"""
    warning = ""
    if model_type == "untrained":
        warning = " ⚠ WARNING: Model is UNTRAINED - predictions will be inaccurate!"
    
    return jsonify({
        'name': 'Sickle Cell Anemia Classifier',
        'version': '1.0',
        'model_type': model_type,
        'status': 'ready' if model is not None else 'not ready',
        'warning': warning
    }), 200

if __name__ == '__main__':
    # Load model on startup
    if not load_or_create_model():
        print("ERROR: Failed to load or create model!")
        exit(1)
    
    # Run Flask app
    print("\n" + "="*50)
    print("Sickle Cell Anemia Classifier Web App")
    print("="*50)
    print("Server running at: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
