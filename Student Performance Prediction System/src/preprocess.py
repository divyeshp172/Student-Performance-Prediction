import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(df, target_col=None):
    """
    Perform complete preprocessing automatically based on the dataset.
    """
    # 1. Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
            
    # 2. Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Target column detection if not provided
    if target_col is None:
        # Assuming the target might be the last column or named somewhat related to 'score' or 'G3'
        potential_targets = [col for col in df.columns if 'score' in col.lower() or col == 'G3' or col == 'target']
        if potential_targets:
            target_col = potential_targets[-1]
        else:
            target_col = df.columns[-1] # Default to last column
            
    # Remove unnecessary columns automatically (e.g. ID like columns with all unique values)
    cols_to_drop = []
    for col in df.columns:
        if df[col].nunique() == len(df) and col != target_col:
            cols_to_drop.append(col)
    
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    # 3. Handle outliers for numerical columns using percentiles
    for col in num_cols:
        lower = X[col].quantile(0.01)
        upper = X[col].quantile(0.99)
        X[col] = X[col].clip(lower, upper)
        
    # 4. Encode categorical columns (One-Hot Encoding)
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Save the feature columns so we can use them in prediction
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
    joblib.dump(X.columns.tolist(), os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_columns.pkl'))
    joblib.dump(cat_cols, os.path.join(os.path.dirname(__file__), '..', 'models', 'cat_cols.pkl'))
    
    # 5. Split dataset 80-20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Scale numerical columns
    scaler = StandardScaler()
    if num_cols:
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # Save scaler and numerical column names
    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl'))
    joblib.dump(num_cols, os.path.join(os.path.dirname(__file__), '..', 'models', 'num_cols.pkl'))
    
    return X_train, X_test, y_train, y_test, target_col
