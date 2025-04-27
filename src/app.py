import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import pickle
import sys
from model_evaluation import (
    load_model_and_features, 
    evaluate_model, 
    plot_predictions, 
    plot_residuals, 
    plot_error_distribution
)

# Configure the page
st.set_page_config(
    page_title="Alloy Hardness Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS with improved text colors for better readability
st.markdown("""
<style>
    /* Main text color for better readability */
    body {
        color: #333333;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Message boxes */
    .success-msg {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
        color: #1B5E20;
        margin-bottom: 1rem;
    }
    .info-msg {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #2196F3;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .warning-msg {
        background-color: #FFF8E1;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FFC107;
        color: #FF6F00;
        margin-bottom: 1rem;
    }
    .error-msg {
        background-color: #FFEBEE;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #F44336;
        color: #B71C1C;
        margin-bottom: 1rem;
    }
    
    /* Labels and text */
    label {
        color: #1E3A8A !important;
        font-weight: 500 !important;
    }
    p {
        color: #333333;
        line-height: 1.6;
    }
    
    /* Data tables */
    .dataframe {
        color: #333333;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #F8F9FA;
    }
    
    /* Buttons */
    .stButton&gt;button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton&gt;button:hover {
        background-color: #0D2B76;
    }
    
    /* Number inputs */
    .stNumberInput input {
        color: #333333 !important;
        font-weight: 500;
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #1E3A8A !important;
    }
    
    /* Prediction result */
    .prediction-result {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
    }
    .prediction-value {
        color: #1E3A8A;
        font-size: 3rem;
        font-weight: bold;
    }
    .prediction-label {
        color: #333333;
        font-size: 1.2rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

def load_model(verbose=False):
    """
    Load the trained model and selected features from various possible locations.
    Returns the model and selected features if successful, None otherwise.
    """
    try:
        # Try multiple possible paths for the model
        model_paths = [
            'models/Hardness_Prediction_Model.pkl',  # This worked in the logs
            'Hardness_Prediction_Model.pkl',
            '../models/Hardness_Prediction_Model.pkl',
            os.path.join(os.path.dirname(__file__), 'models/Hardness_Prediction_Model.pkl'),
            os.path.join(os.path.dirname(__file__), 'Hardness_Prediction_Model.pkl'),
            os.path.join(os.path.dirname(__file__), '../models/Hardness_Prediction_Model.pkl')
        ]
        
        model = None
        model_path_used = None
        for path in model_paths:
            try:
                if verbose:
                    st.write(f"Attempting to load model from: {os.path.abspath(path)}")
                with open(path, 'rb') as file:
                    model = pickle.load(file)
                model_path_used = path
                if verbose:
                    st.write(f"Model loaded successfully from {path}")
                break
            except Exception as e:
                if verbose:
                    st.write(f"Could not load model from {path}: {str(e)}")
                continue
        
        if model is None:
            raise FileNotFoundError("Could not find the model file in any of the expected locations")
        
        # Try multiple possible paths for the features
        feature_paths = [
            'models/Selected_Features.json',  # This worked in the logs
            'models/Selected_Fearures.json',  # Handle the typo in the original filename
            'Selected_Features.json',
            'Selected_Fearures.json',
            '../models/Selected_Features.json',
            '../models/Selected_Fearures.json',
            os.path.join(os.path.dirname(__file__), 'models/Selected_Features.json'),
            os.path.join(os.path.dirname(__file__), 'models/Selected_Fearures.json'),
            os.path.join(os.path.dirname(__file__), 'Selected_Features.json'),
            os.path.join(os.path.dirname(__file__), 'Selected_Fearures.json'),
            os.path.join(os.path.dirname(__file__), '../models/Selected_Features.json'),
            os.path.join(os.path.dirname(__file__), '../models/Selected_Fearures.json')
        ]
        
        selected_features = None
        feature_path_used = None
        for path in feature_paths:
            try:
                if verbose:
                    st.write(f"Attempting to load features from: {os.path.abspath(path)}")
                with open(path, 'r') as f:
                    selected_features = json.load(f)
                feature_path_used = path
                if verbose:
                    st.write(f"Features loaded successfully from {path}")
                break
            except Exception as e:
                if verbose:
                    st.write(f"Could not load features from {path}: {str(e)}")
                continue
        
        if selected_features is None:
            raise FileNotFoundError("Could not find the features file in any of the expected locations")
        
        if not verbose:
            st.markdown(f"&lt;div class=&#39;success-msg&#39;&gt;✅ Model loaded from {model_path_used}&lt;/div&gt;", unsafe_allow_html=True)
            st.markdown(f"&lt;div class=&#39;success-msg&#39;&gt;✅ Features loaded from {feature_path_used}&lt;/div&gt;", unsafe_allow_html=True)
            
        return model, selected_features
    except Exception as e:
        st.markdown(f"&lt;div class=&#39;error-msg&#39;&gt;❌ Error loading model or features: {str(e)}&lt;/div&gt;", unsafe_allow_html=True)
        return None, None

def load_data(verbose=False):
    """
    Load the training and test data from various possible locations.
    Returns the train and test data if successful, None otherwise.
    """
    try:
        # Try multiple possible paths for the data
        data_paths = [
            'data/train_data.csv',  # This worked in the logs
            'train_data.csv',
            '../data/train_data.csv',
            os.path.join(os.path.dirname(__file__), 'data/train_data.csv'),
            os.path.join(os.path.dirname(__file__), 'train_data.csv'),
            os.path.join(os.path.dirname(__file__), '../data/train_data.csv')
        ]
        
        train_data = None
        train_path_used = None
        for path in data_paths:
            try:
                if verbose:
                    st.write(f"Attempting to load train data from: {os.path.abspath(path)}")
                train_data = pd.read_csv(path, encoding='ISO-8859-1')
                train_path_used = path
                if verbose:
                    st.write(f"Train data loaded successfully from {path}")
                break
            except Exception as e:
                if verbose:
                    st.write(f"Could not load train data from {path}: {str(e)}")
                continue
        
        if train_data is None:
            raise FileNotFoundError("Could not find the train data file in any of the expected locations")
        
        if not verbose:
            st.markdown(f"&lt;div class=&#39;success-msg&#39;&gt;✅ Training data loaded from {train_path_used}&lt;/div&gt;", unsafe_allow_html=True)
            
        return train_data
    except Exception as e:
        st.markdown(f"&lt;div class=&#39;error-msg&#39;&gt;❌ Error loading data: {str(e)}&lt;/div&gt;", unsafe_allow_html=True)
        return None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Make Prediction", "Data Exploration", "About"])

# Debug toggle in sidebar
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

if page == "Home":
    st.markdown("&lt;h1 class=&#39;main-header&#39;&gt;Alloy Hardness Prediction System&lt;/h1&gt;", unsafe_allow_html=True)
    
    st.markdown("""
    &lt;div class=&#39;info-msg&#39;&gt;
    &lt;p&gt;Welcome to the Alloy Hardness Prediction System. This application uses machine learning to predict the hardness of alloys based on their composition and processing parameters.&lt;/p&gt;
    &lt;/div&gt;
    """, unsafe_allow_html=True)
    
    st.markdown("&lt;h3 class=&#39;sub-header&#39;&gt;How to use this application:&lt;/h3&gt;", unsafe_allow_html=True)
    st.markdown("""
    1. **Make Prediction**: Enter the composition and processing parameters of your alloy to get a hardness prediction.
    2. **Data Exploration**: Explore the training data and understand the relationships between different features.
    3. **About**: Learn more about the model and the science behind hardness prediction.
    """)
    
    # Load model and data in the background
    with st.spinner("Loading model and data..."):
        model, selected_features = load_model(verbose=debug_mode)
        train_data = load_data(verbose=debug_mode)
    
    if model is not None and train_data is not None:
        st.markdown("&lt;div class=&#39;success-msg&#39;&gt;✅ System is ready for predictions!&lt;/div&gt;", unsafe_allow_html=True)
    else:
        st.markdown("&lt;div class=&#39;error-msg&#39;&gt;❌ Failed to load model or data. Please check the error messages above.&lt;/div&gt;", unsafe_allow_html=True)

elif page == "Make Prediction":
    st.markdown("&lt;h1 class=&#39;main-header&#39;&gt;Make a Hardness Prediction&lt;/h1&gt;", unsafe_allow_html=True)
    
    # Load model and data
    model, selected_features = load_model(verbose=debug_mode)
    
    if model is None or selected_features is None:
        st.markdown("&lt;div class=&#39;error-msg&#39;&gt;❌ Failed to load model or features. Please check the error messages above.&lt;/div&gt;", unsafe_allow_html=True)
    else:
        st.markdown("&lt;div class=&#39;info-msg&#39;&gt;Enter the values for each feature to get a hardness prediction.&lt;/div&gt;", unsafe_allow_html=True)
        
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        # Create input fields for each feature
        user_input = {}
        for i, feature in enumerate(selected_features):
            # Alternate between columns
            if i % 2 == 0:
                with col1:
                    user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")
            else:
                with col2:
                    user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")
        
        # Prediction button
        if st.button("Predict Hardness", key="predict_button"):
            # Create a DataFrame from user input
            input_df = pd.DataFrame([user_input])
            
            # Make prediction
            prediction = model.predict(input_df)
            
            # Display prediction with nice formatting
            st.markdown("&lt;h2 class=&#39;sub-header&#39;&gt;Prediction Result&lt;/h2&gt;", unsafe_allow_html=True)
            st.markdown(f"""
            &lt;div class=&#39;prediction-result&#39;&gt;
                &lt;p class=&#39;prediction-label&#39;&gt;Predicted Hardness (HVN)&lt;/p&gt;
                &lt;p class=&#39;prediction-value&#39;&gt;{prediction[0]:.2f}&lt;/p&gt;
            &lt;/div&gt;
            """, unsafe_allow_html=True)

elif page == "Data Exploration":
    st.markdown("&lt;h1 class=&#39;main-header&#39;&gt;Data Exploration&lt;/h1&gt;", unsafe_allow_html=True)
    
    # Load data
    train_data = load_data(verbose=debug_mode)
    
    if train_data is None:
        st.markdown("&lt;div class=&#39;error-msg&#39;&gt;❌ Failed to load data. Please check the error messages above.&lt;/div&gt;", unsafe_allow_html=True)
    else:
        st.markdown("&lt;div class=&#39;info-msg&#39;&gt;Explore the training data to understand the relationships between different features.&lt;/div&gt;", unsafe_allow_html=True)
        
        # Display data overview
        st.markdown("&lt;h2 class=&#39;sub-header&#39;&gt;Data Overview&lt;/h2&gt;", unsafe_allow_html=True)
        st.write(f"Number of samples: {train_data.shape[0]}")
        st.write(f"Number of features: {train_data.shape[1] - 1}")  # Excluding the target variable
        
        # Display first few rows of the data
        st.markdown("&lt;h2 class=&#39;sub-header&#39;&gt;Sample Data&lt;/h2&gt;", unsafe_allow_html=True)
        st.dataframe(train_data.head())
        
        # Display statistics
        st.markdown("&lt;h2 class=&#39;sub-header&#39;&gt;Statistical Summary&lt;/h2&gt;", unsafe_allow_html=True)
        st.dataframe(train_data.describe())
        
        # Correlation heatmap
        st.markdown("&lt;h2 class=&#39;sub-header&#39;&gt;Correlation Heatmap&lt;/h2&gt;", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = train_data.corr()
        sns.heatmap(correlation, annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Feature vs Target plots
        st.markdown("&lt;h2 class=&#39;sub-header&#39;&gt;Feature vs Hardness&lt;/h2&gt;", unsafe_allow_html=True)
        
        # Let user select a feature to plot against hardness
        feature_to_plot = st.selectbox("Select a feature to plot against Hardness:", 
                                      train_data.columns.drop("Hardness (HVN)"))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=feature_to_plot, y="Hardness (HVN)", data=train_data, ax=ax)
        ax.set_title(f"{feature_to_plot} vs Hardness")
        st.pyplot(fig)

elif page == "About":
    st.markdown("&lt;h1 class=&#39;main-header&#39;&gt;About the Hardness Prediction Model&lt;/h1&gt;", unsafe_allow_html=True)
    
    st.markdown("""
    &lt;div class=&#39;info-msg&#39;&gt;
    &lt;p&gt;This application uses a machine learning model to predict the hardness of alloys based on their composition and processing parameters.&lt;/p&gt;
    &lt;/div&gt;
    """, unsafe_allow_html=True)
    
    st.markdown("### Model Information:")
    st.markdown("""
    - **Model Type**: XGBoost Regressor
    - **Feature Selection**: Recursive Feature Elimination with Cross-Validation (RFECV)
    - **Hyperparameter Tuning**: Grid Search with Cross-Validation
    """)
    
    st.markdown("### About Hardness Prediction:")
    st.markdown("""
    Hardness is a measure of the resistance of a material to localized plastic deformation. It is an important property for many engineering applications, as it can indicate wear resistance, strength, and other mechanical properties.
    
    Predicting hardness based on composition and processing parameters can help in the design of new alloys with desired properties, reducing the need for extensive experimental testing.
    """)
    
    st.markdown("### References:")
    st.markdown("""
    1. XGBoost: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
    2. Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)
    3. Streamlit: [https://streamlit.io/](https://streamlit.io/)
    """)
