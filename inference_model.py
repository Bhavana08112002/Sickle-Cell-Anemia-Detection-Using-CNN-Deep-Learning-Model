import os
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any
from pathlib import Path
import logging

from ..config import settings

logger = logging.getLogger(__name__)

class SickleCellModel:
    """Wrapper class for the Sickle Cell Detection model."""
    
    def __init__(self, model_path: str = None):
        """Initialize the model.
        
        Args:
            model_path: Path to the saved model file. If None, uses the path from settings.
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self.input_size = settings.MODEL_INPUT_SIZE
        self.class_names = {0: "Not Sickle", 1: "Sickle"}
        
    def load_model(self) -> None:
        """Load the trained model from disk."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
                
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("Model loaded successfully")
            
            # Test model is callable
            test_input = np.zeros((1, *self.input_size, 3), dtype=np.float32)
            _ = self.model.predict(test_input)
            logger.info("Model test prediction successful")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image data for model inference.
        
        Args:
            image_data: Binary image data
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Convert bytes to numpy array
            image = tf.image.decode_image(image_data, channels=3)
            
            # Get model's expected input shape (excluding batch dimension)
            if self.model is not None:
                input_shape = self.model.input_shape[1:3]  # Get (height, width)
            else:
                input_shape = self.input_size
            
            # Resize to model's expected input shape
            image = tf.image.resize(image, input_shape)
            
            # Convert to float32 and normalize to [0,1]
            image = tf.cast(image, tf.float32) / 255.0
            
            # Add batch dimension
            image = tf.expand_dims(image, axis=0)
            
            return image.numpy()
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Could not process the image: {str(e)}")
    
    def predict(self, image_data: bytes) -> Dict[str, Any]:
        """Run prediction on an image.
        
        Args:
            image_data: Binary image data
            
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            self.load_model()  # Try to load the model if not loaded
            
            if self.model is None:
                raise RuntimeError("Failed to load the model.")
        
        try:
            # Preprocess image
            image = self.preprocess_image(image_data)
            
            # Get prediction
            prediction = self.model.predict(image, verbose=0)
            
            # Handle different model output formats
            if isinstance(prediction, (list, tuple)) and len(prediction) > 0:
                prediction = prediction[0]  # Take first output if multiple outputs
                
            # Convert to numpy array if not already
            prediction = np.array(prediction)
            
            # Handle binary classification (sigmoid output)
            if prediction.shape[-1] == 1:  # Binary classification
                probability = float(prediction[0][0])
                label = 1 if probability >= 0.5 else 0
                confidence = probability if label == 1 else (1 - probability)
            else:  # Multi-class classification
                # Get the index of the highest probability class
                class_id = int(np.argmax(prediction, axis=-1)[0])
                probability = float(np.max(prediction, axis=-1)[0])
                label = class_id
                confidence = probability
            
            # Ensure label is within class names range
            if label not in self.class_names:
                logger.warning(f"Predicted class {label} not in class names. Using default names.")
                class_name = f"Class {label}"
            else:
                class_name = self.class_names[label]
            
            return {
                "label": class_name,
                "probability": float(probability),
                "confidence": float(confidence),
                "class_id": int(label)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise ValueError(f"Error processing image: {str(e)}")

# Global model instance
model = SickleCellModel()

def get_model() -> SickleCellModel:
    """Get the global model instance."""
    return model

def load_model() -> None:
    """Load the model into the global instance."""
    model.load_model()

def predict_from_bytes(image_bytes: bytes) -> dict:
    """
    Make a prediction from image bytes.
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        Dictionary containing prediction results
    """
    global model
    
    try:
        # Ensure model is loaded
        if model.model is None:
            model.load_model()
            
            # Double check if model loaded successfully
            if model.model is None:
                raise RuntimeError("Failed to load the model.")
        
        # Make prediction
        result = model.predict(image_bytes)
        
        # Ensure all values are JSON serializable
        result = {
            'label': str(result.get('label', 'Unknown')),
            'probability': float(result.get('probability', 0.0)),
            'confidence': float(result.get('confidence', 0.0)),
            'class_id': int(result.get('class_id', -1))
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in predict_from_bytes: {str(e)}", exc_info=True)
        raise ValueError(f"Prediction failed: {str(e)}")
