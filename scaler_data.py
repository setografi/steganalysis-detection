# scaler_data.py

from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

def generate_dummy_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

if __name__ == "__main__":
    X_dummy = np.random.rand(10, 10)  # Generate dummy features for scaler
    scaler = generate_dummy_scaler(X_dummy)
    
    # Ensure the models directory exists
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save scaler using joblib
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    print(f"Dummy scaler saved successfully at {scaler_path}.")
