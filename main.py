from flask import Flask, render_template, request, send_file
from PIL import Image
import numpy as np
import os
import pywt
from scipy.stats import skew, kurtosis

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transform_domain_analysis(img):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    energy = np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2)
    
    threshold = np.mean(energy) * 1.5  # Gunakan threshold dinamis
    has_steganography = energy > threshold
    
    return has_steganography

def statistical_analysis(img):
    img_array = np.array(img)
    mean_value = np.mean(img_array)
    std_dev_value = np.std(img_array)
    skewness_value = skew(img_array.flatten())
    kurtosis_value = kurtosis(img_array.flatten())
    
    # Gunakan threshold yang lebih longgar
    is_stego = (mean_value < 120 or std_dev_value < 30 or
                np.abs(skewness_value) > 0.5 or kurtosis_value > 2.5)
    
    return is_stego

def steganalysis_detection(img_path):
    img = Image.open(img_path).convert('L')
    img_array = np.array(img)

    has_steganography = transform_domain_analysis(img_array)
    is_stego = statistical_analysis(img_array)

    if has_steganography or is_stego:
        return "Stego Image Detected"
    else:
        return "Normal Image"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            detection_result = steganalysis_detection(file_path)
            os.remove(file_path)

            return render_template('index.html', message=detection_result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
