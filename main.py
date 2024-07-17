from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from app.utils import allowed_file, extract_features, save_training_image, UPLOAD_FOLDER
from app.ml_model import load_model, predict_stego, train_model, save_model
from app.dwt_analysis import transform_domain_analysis
from app.statistical_analysis import statistical_analysis
from app.visualization import visualize_features

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tambahkan ini di bagian atas main.py
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
TRAINING_FOLDER = os.path.join(os.path.dirname(__file__), 'training_data')

for folder in [UPLOAD_FOLDER, TRAINING_FOLDER, os.path.join(TRAINING_FOLDER, 'normal'), os.path.join(TRAINING_FOLDER, 'stego')]:
    if not os.path.exists(folder):
        os.makedirs(folder)

model, scaler = load_model()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            img = np.array(Image.open(file_path).convert('L'))
            
            # Perform DWT and Statistical Analysis
            energy = transform_domain_analysis(img)
            mean, std_dev, skewness, kurt = statistical_analysis(img)
            
            features = [energy, mean, std_dev, skewness, kurt]
            
            # Initial detection based on DWT and Statistical Analysis
            # You need to implement this logic based on your threshold or criteria
            initial_prediction = detect_stego(features)
            
            # If model exists, use it for prediction
            if model is not None and scaler is not None:
                ml_prediction = predict_stego(model, scaler, features)
            else:
                ml_prediction = "Model not trained yet"

            return render_template('result.html', 
                                   initial_prediction=initial_prediction,
                                   ml_prediction=ml_prediction, 
                                   filename=filename)

    return render_template('index.html')

@app.route('/confirm', methods=['POST'])
def confirm_result():
    filename = request.form.get('filename')
    image_type = request.form.get('image_type')

    if not filename or not image_type:
        return render_template('index.html', message="Missing form data, please try again.")
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if image_type == 'normal':
        save_training_image(file_path, is_stego=False)
        message = "Image saved as normal image for training"
    elif image_type == 'stego':
        save_training_image(file_path, is_stego=True)
        message = "Image saved as stego image for training"
    else:
        message = "Invalid image type"

    os.remove(file_path)
    return render_template('index.html', message=message)

@app.route('/train')
def train():
    X, y = collect_training_data()
    
    if len(X) < 2:
        return "Not enough samples to train the model. At least 2 samples are required."
    
    visualize_features(X, y)
    
    global model, scaler
    model, scaler = train_model(X, y)
    
    if model is None or scaler is None:
        return "Failed to train the model. Please check your data."
    
    save_model(model, scaler)
    return "Model trained successfully. Feature distribution graph saved as 'feature_distribution.png'."

def collect_training_data():
    X = []
    y = []
    for class_name in ['normal', 'stego']:
        folder = os.path.join(TRAINING_FOLDER, class_name)
        for filename in os.listdir(folder):
            if allowed_file(filename):
                file_path = os.path.join(folder, filename)
                img = np.array(Image.open(file_path).convert('L'))
                features = extract_features(img)
                X.append(features)
                y.append(1 if class_name == 'stego' else 0)
    return np.array(X), np.array(y)

def detect_stego(features):
    energy, mean, std_dev, skewness, kurt = features
    
    # Define your thresholds here based on your analysis
    energy_threshold = 1000  # This is an example value, replace with your actual threshold
    std_dev_threshold = 50   # This is an example value, replace with your actual threshold
    skewness_threshold = 0.5 # This is an example value, replace with your actual threshold
    kurt_threshold = 3       # This is an example value, replace with your actual threshold

    # Implement your detection logic
    if (energy > energy_threshold and 
        std_dev > std_dev_threshold and 
        abs(skewness) > skewness_threshold and 
        kurt > kurt_threshold):
        return "Stego Image Detected"
    else:
        return "Normal Image"

if __name__ == '__main__':
    app.run(debug=True)
