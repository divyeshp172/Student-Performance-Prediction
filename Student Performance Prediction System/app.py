import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.predict import predict_score

# --- Page Config ---
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 1rem;
        color: #ddd;
    }
    h1, h2, h3 {
        color: #fff;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #000;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0, 201, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data & Assets ---
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'student_performance.csv')
    try:
        return pd.read_csv(file_path, sep=None, engine='python')
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def load_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), 'models', 'model_metrics.pkl')
    try:
        return joblib.load(metrics_path)
    except:
        return None

df = load_data()
model_info = load_metrics()

# --- Sidebar ---
st.sidebar.title("🎓 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to:", 
                       ["Prediction", "Dataset Information", "Model Performance", "Developer"])
st.sidebar.markdown("---")
st.sidebar.subheader("About Project")
st.sidebar.info("This project leverages Machine Learning to predict student performance based on multiple demographic and academic factors.")

# --- Helper function for categorizing scores ---
def get_performance_category(score, max_score=20):
    percentage = (score / max_score) * 100 if max_score > 0 else 0
    if percentage >= 80:
        return "Excellent 🌟", "success"
    elif percentage >= 60:
        return "Good 👍", "info"
    elif percentage >= 40:
        return "Average 😐", "warning"
    else:
        return "Needs Improvement 📉", "error"

# --- Page: Prediction ---
if page == "Prediction":
    st.title("🎓 Student Performance Prediction System")
    st.markdown("Enter the details below to predict the estimated exam score.")
    
    if df.empty or model_info is None:
        st.error("Model or dataset not found. Please run the training script first (`python src/train.py`).")
    else:
        # Generate form based on actual dataset columns
        target_col = model_info.get('target_col', df.columns[-1])
        features = [col for col in df.columns if col != target_col]
        
        st.markdown("### Input Features")
        
        # Create columns for layout
        cols = st.columns(3)
        input_data = {}
        
        for i, col in enumerate(features):
            with cols[i % 3]:
                if df[col].dtype == 'object' or df[col].nunique() < 10:
                    options = df[col].dropna().unique().tolist()
                    input_data[col] = st.selectbox(f"{col}", options, key=col)
                else:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())
                    input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val, key=col)
                    
        st.markdown("---")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            predict_btn = st.button("🔮 Predict Exam Score", use_container_width=True)
            
        if predict_btn:
            with st.spinner("Analyzing data and generating prediction..."):
                prediction = predict_score(input_data)
                
            if prediction is not None:
                st.markdown("---")
                st.subheader("Results")
                
                # Assume max score is max in target col of dataset, or fallback to 20 or 100
                max_possible = df[target_col].max()
                if max_possible <= 20: max_possible = 20
                elif max_possible <= 100: max_possible = 100
                else: max_possible = max(100, prediction)
                
                category, state = get_performance_category(prediction, max_possible)
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Estimated Exam Score</div>
                            <div class="metric-value">{prediction:.2f} <span style="font-size:1rem; color:#aaa;">/ {max_possible}</span></div>
                        </div>
                    """, unsafe_allow_html=True)
                with res_col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Performance Category</div>
                            <div class="metric-value">{category}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                st.progress(min(1.0, max(0.0, prediction / max_possible)))

# --- Page: Dataset Information ---
elif page == "Dataset Information":
    st.title("📊 Dataset Information")
    if df.empty:
        st.warning("Dataset not found!")
    else:
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Dataset Shape (Rows, Columns)</div>
                    <div class="metric-value">{df.shape[0]} × {df.shape[1]}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            total_missing = df.isnull().sum().sum()
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Missing Values</div>
                    <div class="metric-value">{total_missing}</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### Correlation Heatmap (Numerical Features)")
        num_df = df.select_dtypes(include=['float64', 'int64'])
        if not num_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            sns.heatmap(num_df.corr(), annot=True, cmap='viridis', fmt='.2f', ax=ax,
                       cbar_kws={'label': 'Correlation Coefficient'})
            ax.tick_params(colors='white')
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            st.pyplot(fig)
        else:
            st.info("No numerical features available for correlation heatmap.")

# --- Page: Model Performance ---
elif page == "Model Performance":
    st.title("📈 Model Performance")
    if model_info is None:
        st.warning("Model metrics not found. Please train the models first.")
    else:
        best_name = model_info['best_name']
        metrics = model_info['metrics']
        all_results = model_info['all_results']
        
        st.subheader("🏆 Best Algorithm Selected")
        st.markdown(f"### <span style='color:#00C9FF'>{best_name}</span>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MAE</div>
                    <div class="metric-value">{metrics['MAE']:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value">{metrics['RMSE']:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">R² Score</div>
                    <div class="metric-value">{metrics['R2']:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### Algorithm Comparison")
        # Format the dataframe for better display
        styled_df = all_results.style.highlight_max(subset=['R2'], color='green').highlight_min(subset=['MAE', 'RMSE', 'MSE'], color='green')
        st.dataframe(styled_df, use_container_width=True)
        
        # Simple Bar Chart for R2 Scores
        st.markdown("### R² Score Comparison Chart")
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        sns.barplot(x=all_results.index, y=all_results['R2'], ax=ax, palette='viridis')
        ax.set_ylabel('R² Score', color='white')
        ax.tick_params(colors='white')
        plt.xticks(rotation=45, color='white')
        st.pyplot(fig)

# --- Page: Developer ---
elif page == "Developer":
    st.title("💻 Developer")
    st.markdown("---")
    st.markdown("### Project Name")
    st.info("**Student Performance Prediction System**")
    
    st.markdown("### Tech Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- Python")
        st.markdown("- Pandas")
        st.markdown("- NumPy")
        st.markdown("- Scikit-learn")
    with col2:
        st.markdown("- Matplotlib & Seaborn")
        st.markdown("- Streamlit")
        st.markdown("- XGBoost")
        st.markdown("- Joblib")
        
    st.markdown("### Machine Learning Algorithms Used")
    st.success("""
    1. Linear Regression
    2. Decision Tree Regressor
    3. Random Forest Regressor (Optimized with GridSearchCV)
    4. Gradient Boosting Regressor
    5. XGBoost
    """)
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit.")
