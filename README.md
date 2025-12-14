# Sickle Cell Anemia Classifier - Web Application

A professional AI-powered web application for classifying blood cell images to detect sickle cell anemia using deep learning models.

## Features

✨ **Modern Web Interface**
- Drag-and-drop image upload
- Real-time image preview
- Beautiful, responsive design
- Mobile-friendly layout

🔬 **Smart Classification**
- Deep learning-based image classification
- Confidence score display
- Detailed analysis results
- Support for multiple image formats

⚡ **Performance**
- Fast inference with optimized models
- Batch processing support
- Progress indicators
- Error handling and validation

## Project Structure

```
Bhavana/
├── Sickle_cell_anemia.py    # Original training script
├── app.py                    # Flask backend server
├── index.html                # Frontend HTML
├── style.css                 # Styling
├── script.js                 # Frontend JavaScript
├── requirements.txt          # Python dependencies
├── model_checkpoint.h5       # Trained model (if available)
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Trained Model (Optional)

If you have a trained model, save it as `model_checkpoint.h5` in the project directory:

```python
# After training in Sickle_cell_anemia.py
model.save('model_checkpoint.h5')
```

If no model is provided, the app will create an untrained model for testing purposes.

### Step 3: Run the Application

```bash
python app.py
```

The application will start at: **http://localhost:5000**

## Usage

1. **Open the Web App**: Navigate to `http://localhost:5000` in your browser
2. **Upload Image**: Click "Choose File" or drag-and-drop a blood cell image
3. **Preview**: See your image in the preview area
4. **Classify**: Click "Classify Image" to get predictions
5. **View Results**: See classification, confidence score, and analysis
6. **Reset**: Click "Reset" to upload another image

## Model Information

### Available Models

The application uses **Model 2** (Convolutional Neural Network with Batch Normalization and Dropout):

```
Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
Dense(256) → Dropout(0.5)
Dense(1, sigmoid)
```

### Classification Output

- **Sickle Cell**: Indicates presence of sickle cell anemia characteristics
- **Normal**: Indicates normal blood cells
- **Confidence**: Percentage confidence in the prediction

## Configuration

### Update Dataset Path

Edit `Sickle_cell_anemia.py` line 22:

```python
data_dir = r"C:\Your\Dataset\Path\Here"
```

### Training the Model

Run the original training script to generate a trained model:

```bash
python Sickle_cell_anemia.py
```

This will create a `model_checkpoint.h5` file that the web app will automatically use.

### Adjust Flask Settings

In `app.py`, you can modify:

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

- `debug=True`: Enable debug mode (disable in production)
- `host='0.0.0.0'`: Allow external connections
- `port=5000`: Change port number if needed

## API Endpoints

### POST /classify
Classify an uploaded blood cell image.

**Request:**
```
Content-Type: multipart/form-data
Body: image (file)
```

**Response:**
```json
{
  "classification": "Sickle Cell",
  "confidence": 0.92,
  "model_type": "loaded",
  "raw_prediction": 0.92
}
```

### GET /health
Check application health status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "loaded"
}
```

### GET /info
Get application information.

**Response:**
```json
{
  "name": "Sickle Cell Anemia Classifier",
  "version": "1.0",
  "model_type": "loaded",
  "status": "ready"
}
```

## Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- Bitmap (.bmp)
- TIFF (.tiff)
- Maximum file size: 10MB

## Troubleshooting

### Port Already in Use

If port 5000 is already in use, change it in `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Model Not Loading

Ensure `model_checkpoint.h5` is in the same directory as `app.py`, or train a new model:

```bash
python Sickle_cell_anemia.py
```

### Dependencies Installation Issues

Try installing TensorFlow separately:

```bash
pip install --upgrade tensorflow
```

### CUDA/GPU Issues

If you encounter GPU-related errors, use CPU-only installation:

```bash
pip install tensorflow-cpu
```

## Performance Tips

1. **Image Quality**: Use high-quality blood cell images for better predictions
2. **Image Size**: Images are resized to 256×256, so clarity matters
3. **Model Training**: Train the model with your complete dataset for best results
4. **Browser**: Use modern browsers (Chrome, Firefox, Safari, Edge) for best UI experience

## Future Enhancements

- [ ] Batch processing multiple images
- [ ] Model comparison dashboard
- [ ] Data export and reporting
- [ ] Historical prediction tracking
- [ ] Multi-model ensemble predictions
- [ ] Mobile app version

## Medical Disclaimer

This application is for educational and research purposes only. **It is not intended for clinical diagnosis or treatment decisions.** Always consult with qualified medical professionals for diagnosis and treatment of sickle cell anemia or any medical condition.

## License

This project is provided as-is for educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Ensure all dependencies are installed correctly
3. Verify the dataset path is correct
4. Check that the model file exists (if using a trained model)

## Credits

- Deep Learning: TensorFlow/Keras
- Web Framework: Flask
- Frontend: HTML5, CSS3, JavaScript
- Medical Dataset: Blood Cell Images

---

**Created**: November 2025  
**Version**: 1.0
