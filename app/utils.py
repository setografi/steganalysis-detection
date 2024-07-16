import os
from PIL import Image
import numpy as np
from .dwt_analysis import transform_domain_analysis
from .statistical_analysis import statistical_analysis

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
TRAINING_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training_data')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(img):
    energy = transform_domain_analysis(img)
    mean, std_dev, skewness, kurt = statistical_analysis(img)
    return [energy, mean, std_dev, skewness, kurt]

def save_training_image(file_path, is_stego):
    img = Image.open(file_path)
    folder = os.path.join(TRAINING_FOLDER, 'stego' if is_stego else 'normal')
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.basename(file_path)
    save_path = os.path.join(folder, filename)
    img.save(save_path)
    print(f"Saved training image to {save_path}")