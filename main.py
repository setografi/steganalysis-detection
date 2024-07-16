from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from app.utils import allowed_file, extract_features, save_training_image, UPLOAD_FOLDER
from app.ml_model import load_model, predict_stego, train_model, save_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
            features = extract_features(img)
            
            if model is None or scaler is None:
                prediction = "Model not trained yet"
            else:
                prediction = predict_stego(model, scaler, features)

            return render_template('result.html', message=prediction, filename=filename)

    return render_template('index.html')

@app.route('/confirm', methods=['POST'])
def confirm_result():
    filename = request.form['filename']
    is_correct = request.form['is_correct'] == 'yes'
    prediction = request.form['prediction']

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if is_correct:
        is_stego = prediction == "Stego Image Detected"
        save_training_image(file_path, is_stego)
        message = "Image saved for training"
    else:
        message = "Image discarded"

    os.remove(file_path)
    return render_template('index.html', message=message)

@app.route('/train')
def train():
    # Implementasi logika untuk melatih ulang model
    # Anda perlu mengumpulkan semua data pelatihan, mengekstrak fitur, dan melatih model
    # Contoh sederhana:
    X, y = collect_training_data()  # Anda perlu mengimplementasikan fungsi ini
    model, scaler = train_model(X, y)
    save_model(model, scaler)
    return "Model trained successfully"

def collect_training_data():
    # Implementasi fungsi ini untuk mengumpulkan data pelatihan
    # dari folder training_data
    pass

if __name__ == '__main__':
    app.run(debug=True)