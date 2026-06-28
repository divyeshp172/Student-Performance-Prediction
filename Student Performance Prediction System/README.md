# 🎓 Student Performance Prediction System

## Project Description
This project is an end-to-end Machine Learning solution designed to predict student performance (exam scores) based on demographic, social, and academic background features. It includes comprehensive data preprocessing, exploratory data analysis, and the training of multiple regression models, alongside an interactive Streamlit application to visualize the results and make real-time predictions.

## Folder Structure
```
Student-Performance-Prediction/
│
├── data/
│   └── student_performance.csv     # Dataset file
│
├── models/
│   └── student_model.pkl           # Trained Random Forest model
│
├── notebooks/
│   └── EDA.ipynb                   # Exploratory Data Analysis Notebook
│
├── src/
│   ├── preprocess.py               # Data cleaning and preprocessing functions
│   ├── train.py                    # Model training, evaluation, and saving
│   ├── predict.py                  # Functions to make predictions using the trained model
│   └── utils.py                    # Helper functions (loading data, saving models, etc.)
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignored files
```

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone https://github.com/your-username/Student-Performance-Prediction.git
   cd Student-Performance-Prediction
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. **Place the dataset**: Ensure `student_performance.csv` is placed inside the `data/` folder.
2. **Train the Model**: Run the training script to preprocess data, evaluate algorithms, and save the best model.
   ```bash
   python src/train.py
   ```
3. **Launch the Web App**: Start the Streamlit application.
   ```bash
   streamlit run app.py
   ```

## Features
- **Exploratory Data Analysis**: Visualizations for data distribution, missing values, and feature correlation.
- **Data Preprocessing**: Handling of missing values, outliers, automated categorical encoding, and numerical scaling.
- **Model Evaluation**: Compares Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost using MAE, MSE, RMSE, and R² scores.
- **Interactive UI**: A modern Streamlit app offering real-time prediction, dataset exploration, and model performance metrics.
- **Dynamic Input**: The prediction page automatically adapts to the actual dataset columns provided.

## Algorithms Used
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor (with GridSearchCV Hyperparameter Tuning)
- Gradient Boosting Regressor
- XGBoost Regressor

## Future Improvements
- **Deep Learning**: Integration of Neural Networks for potentially higher accuracy.
- **More Features**: Incorporate textual feedback from teachers using NLP techniques.
- **Deployment**: Deploy the app on cloud platforms like AWS, Heroku, or Streamlit Cloud.
- **Database Integration**: Save predictions to a database for long-term tracking.
