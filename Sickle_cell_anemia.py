"""
Sickle Cell Anemia Detection - Model Training and Evaluation
----------------------------------------------------------
This script trains and evaluates multiple CNN models for detecting sickle cells in blood smear images.
The best performing model is saved for deployment.

Author: Your Name
Date: 2025-11-26
"""

import os
import json
import time
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    BatchNormalization, Input
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path
import pandas as pd


def load_and_preprocess_data(data_dir, img_size=(256, 256), batch_size=32, validation_split=0.2, test_split=0.1):
    """
    Load and preprocess the dataset.
    
    Args:
        data_dir: Path to the dataset directory
        img_size: Tuple of (height, width) for resizing images
        batch_size: Batch size for the dataset
        validation_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        
    Returns:
        train, val, test: TensorFlow datasets
        class_names: List of class names
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")
    
    print(f"Loading dataset from: {data_dir}")
    
    # Load the full dataset
    full_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    # Get class names
    class_names = full_dataset.class_names
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Normalize pixel values
    full_dataset = full_dataset.map(lambda x, y: (x/255.0, y))
    
    # Calculate dataset sizes
    dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
    test_size = int(dataset_size * test_split)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size - test_size
    
    # Split the dataset
    train = full_dataset.take(train_size)
    val = full_dataset.skip(train_size).take(val_size)
    test = full_dataset.skip(train_size + val_size).take(test_size)
    
    print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test batches")
    
    return train, val, test, class_names

def create_callbacks(model_name, patience=5, min_lr=1e-6):
    """Create training callbacks.
    
    Args:
        model_name: Name of the model for saving checkpoints
        patience: Patience for early stopping
        min_lr: Minimum learning rate for reduction
        
    Returns:
        List of callbacks
    """
    # Create output directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Model checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        f'checkpoints/{model_name}_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=patience // 2,
        min_lr=min_lr,
        verbose=1
    )
    
    return [checkpoint, early_stopping, reduce_lr]

def create_model1(input_shape=(256, 256, 3)):
    """Baseline CNN model."""
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(16, (3,3), activation='relu', padding='same'),
        MaxPooling2D(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(),
        Conv2D(16, (3,3), activation='relu', padding='same'),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name='Baseline_CNN')
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


def create_model2(input_shape=(256, 256, 3)):
    """Deeper CNN with BatchNorm and Dropout."""
    model = Sequential([
        Input(shape=input_shape),
        
        # First conv block
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        # Second conv block
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        # Third conv block
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        # Dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name='Deeper_CNN')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


def create_model3(input_shape=(256, 256, 3)):
    """Smaller CNN for faster training."""
    model = Sequential([
        Input(shape=input_shape),
        
        # First conv block
        Conv2D(8, (3,3), activation='relu', padding='same'),
        MaxPooling2D(),
        
        # Second conv block
        Conv2D(16, (3,3), activation='relu', padding='same'),
        MaxPooling2D(),
        Dropout(0.25),

        # Dense layers
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ], name='Small_CNN')
    
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


def create_model4(input_shape=(256, 256, 3)):
    """Transfer learning model using MobileNetV2."""
    # Load pre-trained MobileNetV2 without the top classification layer
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'  # GlobalAveragePooling2D instead of Flatten
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Build the complete model
    inputs = Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs, name='MobileNetV2_Transfer')
    
    # Use a lower learning rate for transfer learning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


def train_model(model_creator, train_data, val_data, epochs=20, batch_size=32):
    """Train a model and return the history and the trained model."""
    model = model_creator()
    model_name = model.name
    
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    # Create callbacks
    callbacks = create_callbacks(model_name)
    
    # Train the model
    start_time = time.time()
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    return model, history, training_time

def evaluate_model(model, test_data, class_names):
    """Evaluate the model on test data and return metrics."""
    print(f"\nEvaluating {model.name} on test data...")
    
    # Get predictions
    y_true = []
    y_pred_proba = []
    
    for x_batch, y_batch in test_data:
        y_true.extend(y_batch.numpy())
        y_pred_proba.extend(model.predict(x_batch, verbose=0).flatten())
    
    y_true = np.array(y_true)
    y_pred = (np.array(y_pred_proba) > 0.5).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred_proba)
    }
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model.name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('evaluation', exist_ok=True)
    plt.savefig(f'evaluation/confusion_matrix_{model.name}.png')
    plt.close()
    
    return metrics

def save_model_artifacts(model, history, metrics, training_time, output_dir='saved_models'):
    """Save model, training history, and metrics."""
    model_name = model.name
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, 'model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save the training history
    history_path = os.path.join(model_dir, 'history.json')
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        json.dump(history_dict, f, indent=2)
    
    # Save metrics
    metrics['training_time_seconds'] = training_time
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Training history and metrics saved to {model_dir}")
    
    return model_path

def plot_training_history(history, model_name):
    """Plot training history for a single model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('evaluation', exist_ok=True)
    plt.savefig(f'evaluation/training_history_{model_name}.png')
    plt.close()

def compare_models(metrics_dict):
    """Compare performance metrics across models."""
    metrics_df = pd.DataFrame(metrics_dict).T
    
    # Plot metrics comparison
    plt.figure(figsize=(12, 6))
    metrics_df[['accuracy', 'precision', 'recall', 'f1']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('evaluation/model_comparison.png')
    plt.close()
    
    return metrics_df

def predict_single_image(model, image_path, target_size=(256, 256)):
    """Make a prediction on a single image."""
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=target_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    class_idx = 1 if prediction > 0.5 else 0
    confidence = prediction if class_idx == 1 else (1 - prediction)
    
    return {
        'class_idx': class_idx,
        'probability': float(prediction),
        'confidence': float(confidence),
        'class_name': 'Sickle' if class_idx == 1 else 'Not Sickle'
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate Sickle Cell Anemia detection models.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Only evaluate existing models without training')
    args = parser.parse_args()
    
    # Load and preprocess the data
    train_data, val_data, test_data, class_names = load_and_preprocess_data(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # Model creators to evaluate
    model_creators = [
        create_model1,
        create_model2,
        create_model3,
        create_model4
    ]
    
    # Dictionary to store metrics for all models
    all_metrics = {}
    
    # Train and evaluate each model
    for model_creator in model_creators:
        model_name = model_creator().name
        
        if not args.evaluate:
            # Train the model
            model, history, training_time = train_model(
                model_creator,
                train_data,
                val_data,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            # Plot training history
            plot_training_history(history, model_name)
        else:
            # Load the pre-trained model
            model_path = os.path.join('saved_models', model_name, 'model.h5')
            if not os.path.exists(model_path):
                print(f"Model {model_name} not found at {model_path}. Skipping...")
                continue
                
            model = load_model(model_path, compile=True)
            print(f"Loaded pre-trained model: {model_name}")
        
        # Evaluate the model
        metrics = evaluate_model(model, test_data, class_names)
        all_metrics[model_name] = metrics
        
        if not args.evaluate:
            # Save model artifacts
            save_model_artifacts(model, history, metrics, training_time)
    
    # Compare all models
    if all_metrics:
        metrics_df = compare_models(all_metrics)
        print("\nModel Comparison:")
        print(metrics_df)
        
        # Save comparison results
        metrics_df.to_csv('evaluation/model_comparison.csv')
        print("\nComparison results saved to evaluation/model_comparison.csv")
    
    print("\nDone!")

if __name__ == "__main__":
    # Ensure TensorFlow uses GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"Using GPU: {physical_devices[0]}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    main()
