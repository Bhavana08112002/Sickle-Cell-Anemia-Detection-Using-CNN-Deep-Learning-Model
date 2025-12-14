// DOM Elements
const imageInput = document.getElementById('imageInput');
const browseBtn = document.getElementById('browseBtn');
const previewImage = document.getElementById('previewImage');
const previewArea = document.getElementById('previewArea');
const classifyBtn = document.getElementById('classifyBtn');
const resetBtn = document.getElementById('resetBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');
const resultsContent = document.getElementById('resultsContent');
const resultsArea = document.getElementById('resultsArea');
const uploadBox = document.querySelector('.upload-box');

let selectedFile = null;

// Check model status on page load
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/info');
        const data = await response.json();
        
        if (data.model_type === 'untrained') {
            document.getElementById('warningBanner').style.display = 'block';
        }
    } catch (error) {
        console.error('Error checking model status:', error);
    }
});

// Browse button click
browseBtn.addEventListener('click', () => {
    imageInput.click();
});

// File input change
imageInput.addEventListener('change', handleFileSelect);

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
    uploadBox.style.backgroundColor = 'rgba(118, 75, 162, 0.1)';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.backgroundColor = 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        handleFileSelect();
    }
});

// Handle file selection
function handleFileSelect() {
    const file = imageInput.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }

    selectedFile = file;
    clearError();
    displayPreview(file);
    showClassifyButton();
}

// Display preview
function displayPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        previewArea.querySelector('.placeholder').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Show classify button
function showClassifyButton() {
    classifyBtn.style.display = 'block';
    resetBtn.style.display = 'block';
}

// Classify image
classifyBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }

    const formData = new FormData();
    formData.append('image', selectedFile);

    showLoading(true);
    clearError();

    try {
        const response = await fetch('/classify', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
        }
    } catch (error) {
        showError('Failed to classify image: ' + error.message);
        console.error('Error:', error);
    } finally {
        showLoading(false);
    }
});

// Display results
function displayResults(data) {
    const classification = data.classification;
    const confidence = (data.confidence * 100).toFixed(2);

    document.getElementById('classification').textContent = classification;
    document.getElementById('confidence').textContent = confidence + '%';
    document.getElementById('confidenceBar').style.width = confidence + '%';

    // Set color based on classification
    const resultValue = document.getElementById('classification');
    if (classification === 'Sickle Cell') {
        resultValue.style.color = '#dc2626';
    } else {
        resultValue.style.color = '#16a34a';
    }

    // Set progress bar color
    const confidenceBar = document.getElementById('confidenceBar');
    if (confidence >= 80) {
        confidenceBar.style.background = 'linear-gradient(90deg, #16a34a 0%, #15803d 100%)';
    } else if (confidence >= 60) {
        confidenceBar.style.background = 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)';
    } else {
        confidenceBar.style.background = 'linear-gradient(90deg, #dc2626 0%, #b91c1c 100%)';
    }

    // Show detailed results
    let details = `<strong>Analysis:</strong><br>`;
    if (classification === 'Sickle Cell') {
        details += `This image shows characteristics consistent with sickle cell anemia. `;
        details += `The model is ${confidence}% confident in this classification. `;
        details += `<br><br><strong>Recommendation:</strong> Further medical evaluation is recommended.`;
    } else {
        details += `This image shows normal blood cells. `;
        details += `The model is ${confidence}% confident in this classification. `;
        details += `<br><br><strong>Note:</strong> This is an AI prediction and should be confirmed by a medical professional.`;
    }

    document.getElementById('resultDetails').innerHTML = details;

    // Show results
    resultsArea.querySelector('.placeholder').style.display = 'none';
    resultsContent.style.display = 'block';
}

// Show loading state
function showLoading(show) {
    loadingSpinner.style.display = show ? 'flex' : 'none';
    classifyBtn.disabled = show;
    if (show) {
        classifyBtn.textContent = 'Classifying...';
    } else {
        classifyBtn.textContent = 'Classify Image';
    }
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

// Clear error message
function clearError() {
    errorMessage.textContent = '';
    errorMessage.style.display = 'none';
}

// Reset form
resetBtn.addEventListener('click', () => {
    imageInput.value = '';
    previewImage.src = '';
    previewImage.style.display = 'none';
    previewArea.querySelector('.placeholder').style.display = 'block';
    resultsContent.style.display = 'none';
    resultsArea.querySelector('.placeholder').style.display = 'block';
    classifyBtn.style.display = 'none';
    resetBtn.style.display = 'none';
    clearError();
    selectedFile = null;
});
