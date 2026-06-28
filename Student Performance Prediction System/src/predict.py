import os
import joblib
import pandas as pd
from .utils import load_model

def predict_score(input_data):
    """
    Accepts a dictionary of input features and returns the predicted score.
    """
    # Paths
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, '..', 'models', 'student_model.pkl')
    scaler_path = os.path.join(base_dir, '..', 'models', 'scaler.pkl')
    num_cols_path = os.path.join(base_dir, '..', 'models', 'num_cols.pkl')
    feature_cols_path = os.path.join(base_dir, '..', 'models', 'feature_columns.pkl')
    cat_cols_path = os.path.join(base_dir, '..', 'models', 'cat_cols.pkl')
    
    # Load assets
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        num_cols = joblib.load(num_cols_path)
        feature_columns = joblib.load(feature_cols_path)
        cat_cols = joblib.load(cat_cols_path)
    except Exception as e:
        print(f"Error loading model or preprocessors: {e}")
        return None
        
    # Convert input to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Preprocess categorical (dummy variables)
    # We need to make sure we use the same categories
    # Instead of just get_dummies, we create a dummy dataframe with the feature columns
    
    if cat_cols:
        df_input = pd.get_dummies(df_input, columns=[c for c in cat_cols if c in df_input.columns])
        
    # Ensure all required columns are present
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
            
    # Reorder columns to match training data
    df_input = df_input[feature_columns]
    
    # Scale numerical columns
    if num_cols:
        df_input[num_cols] = scaler.transform(df_input[num_cols])
    
    # Predict
    prediction = model.predict(df_input)[0]
    return prediction
