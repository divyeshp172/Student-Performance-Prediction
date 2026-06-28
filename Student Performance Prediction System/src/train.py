import os
import pandas as pd
from preprocess import preprocess_data
from utils import load_data, save_model, evaluate_model
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def main():
    print("Loading data...")
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'student_performance.csv')
    df = load_data(file_path)
    
    if df is None:
        print("Failed to load data.")
        return
        
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, target_col = preprocess_data(df)
    print(f"Target column detected as: {target_col}")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    }
    
    print("Training models...")
    results = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        results[name] = metrics
        
        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_model_name = name
            best_model = model
            
    print("\n--- Model Evaluation ---")
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    print(f"\nBest Initial Model: {best_model_name} with R2: {best_r2:.4f}")
    
    print("\nPerforming Hyperparameter Tuning for Random Forest...")
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    rf_metrics = evaluate_model(y_test, y_pred_rf)
    
    print(f"Tuned Random Forest R2: {rf_metrics['R2']:.4f}")
    
    # Save the best model overall
    if rf_metrics['R2'] > best_r2:
        final_model = best_rf
        final_model_name = 'Tuned Random Forest'
        final_metrics = rf_metrics
    else:
        final_model = best_model
        final_model_name = best_model_name
        final_metrics = results[best_model_name]
        
    print(f"Saving {final_model_name} as the best model...")
    save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'student_model.pkl')
    save_model(final_model, save_path)
    
    # Save results to display in app
    joblib.dump({
        'best_name': final_model_name,
        'metrics': final_metrics,
        'all_results': results_df,
        'target_col': target_col
    }, os.path.join(os.path.dirname(__file__), '..', 'models', 'model_metrics.pkl'))
    print("Training complete!")
    
if __name__ == "__main__":
    main()
