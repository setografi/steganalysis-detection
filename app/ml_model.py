from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np

def train_model(X, y):
    if len(X) < 2:
        print("Not enough samples to train the model. At least 2 samples are required.")
        return None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if len(X) < 5:  # If we have very few samples, don't split the data
        model = SVC(kernel='rbf', C=1.0)
        model.fit(X_scaled, y)
        accuracy = model.score(X_scaled, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = SVC(kernel='rbf', C=1.0)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
    
    print(f"Model accuracy: {accuracy}")
    
    return model, scaler

def save_model(model, scaler, model_path='model.joblib', scaler_path='scaler.joblib'):
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, os.path.join(model_dir, model_path))
    joblib.dump(scaler, os.path.join(model_dir, scaler_path))

def load_model(model_path='model.joblib', scaler_path='scaler.joblib'):
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(model_dir, model_path)
    scaler_path = os.path.join(model_dir, scaler_path)
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        return None, None

def predict_stego(model, scaler, features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return "Stego Image Detected" if prediction[0] == 1 else "Normal Image"