# model_data.py

from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import os

def generate_dummy_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    # Load dummy data
    training_data_dir = 'training_data'
    X_path = os.path.join(training_data_dir, 'X_dummy.npy')
    y_path = os.path.join(training_data_dir, 'y_dummy.npy')
    
    X = np.load(X_path)
    y = np.load(y_path)
    
    model = generate_dummy_model(X, y)
    
    # Ensure the models directory exists
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model using joblib
    model_path = os.path.join(models_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    print(f"Dummy model saved successfully at {model_path}.")
