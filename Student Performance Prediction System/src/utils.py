import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def load_data(file_path):
    """Load dataset automatically handling separator."""
    try:
        # Use python engine with sep=None to automatically detect delimiter
        df = pd.read_csv(file_path, sep=None, engine='python')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def save_model(model, file_path):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """Load the trained model from disk."""
    if os.path.exists(file_path):
        return joblib.load(file_path)
    else:
        print(f"Model not found at {file_path}")
        return None

def evaluate_model(y_true, y_pred):
    """Evaluate model and return metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
