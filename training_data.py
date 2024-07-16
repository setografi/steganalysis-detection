# training_data.py

import numpy as np
import os

def generate_dummy_data():
    # Generate dummy features X
    X_normal = np.random.rand(100, 10)  # 100 data points with 10 features each for normal images
    X_stego = np.random.rand(100, 10)   # 100 data points with 10 features each for stego images
    
    # Generate dummy labels y
    y_normal = np.zeros(100)    # Labels for normal images (0 for normal)
    y_stego = np.ones(100)      # Labels for stego images (1 for stego)
    
    # Concatenate normal and stego data
    X = np.concatenate((X_normal, X_stego), axis=0)
    y = np.concatenate((y_normal, y_stego))
    
    return X, y

if __name__ == "__main__":
    X, y = generate_dummy_data()
    
    # Ensure the training_data directory exists
    training_data_dir = 'training_data'
    os.makedirs(training_data_dir, exist_ok=True)
    
    # Save dummy data
    X_path = os.path.join(training_data_dir, 'X_dummy.npy')
    y_path = os.path.join(training_data_dir, 'y_dummy.npy')
    np.save(X_path, X)
    np.save(y_path, y)
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Dummy data saved successfully at {training_data_dir}.")
