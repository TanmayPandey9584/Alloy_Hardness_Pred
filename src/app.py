import os
import streamlit as st
import pandas as pd
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

# Add the current directory to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page configuration
st.set_page_config(
    page_title="Hardness Prediction Model",
    page_icon="🔧",
    layout="wide"
)

def load_model():
    """
    Load the trained model and selected features from various possible locations.
    Returns the model and selected features if successful, None otherwise.
    """
    try:
        # Try multiple possible paths for the model
        model_paths = [
            'Hardness_Prediction_Model.pkl',
            '../models/Hardness_Prediction_Model.pkl',
            'models/Hardness_Prediction_Model.pkl',
            os.path.join(os.path.dirname(__file__), 'Hardness_Prediction_Model.pkl'),
            os.path.join(os.path.dirname(__file__), '../models/Hardness_Prediction_Model.pkl'),
            os.path.join(os.path.dirname(__file__), 'models/Hardness_Prediction_Model.pkl')
        ]
        
        model = None
        for path in model_paths:
            try:
                st.write(f"Attempting to load model from: {os.path.abspath(path)}")
                with open(path, 'rb') as file:
                    model = pickle.load(file)
                st.write(f"Model loaded successfully from {path}")
                break
            except Exception as e:
                st.write(f"Could not load model from {path}: {str(e)}")
                continue
        
        if model is None:
            raise FileNotFoundError("Could not find the model file in any of the expected locations")
        
        # Try multiple possible paths for the features
        feature_paths = [
            'Selected_Features.json',
            '../models/Selected_Features.json',
            'models/Selected_Features.json',
            'Selected_Fearures.json',  # Handle the typo in the original filename
            '../models/Selected_Fearures.json',
            'models/Selected_Fearures.json',
            os.path.join(os.path.dirname(__file__), 'Selected_Features.json'),
            os.path.join(os.path.dirname(__file__), '../models/Selected_Features.json'),
            os.path.join(os.path.dirname(__file__), 'models/Selected_Features.json'),
            os.path.join(os.path.dirname(__file__), 'Selected_Fearures.json'),
            os.path.join(os.path.dirname(__file__), '../models/Selected_Fearures.json'),
            os.path.join(os.path.dirname(__file__), 'models/Selected_Fearures.json')
        ]
        
        selected_features = None
        for path in feature_paths:
            try:
                st.write(f"Attempting to load features from: {os.path.abspath(path)}")
                with open(path, 'r') as f:
                    selected_features = json.load(f)
                st.write(f"Features loaded successfully from {path}")
                break
            except Exception as e:
                st.write(f"Could not load features from {path}: {str(e)}")
                continue
        
        if selected_features is None:
            raise FileNotFoundError("Could not find the features file in any of the expected locations")
        
        return model, selected_features
    except Exception as e:
        st.error(f"Error loading model or features: {str(e)}")
        return None, None

def load_data():
    """
    Load the training and test data from various possible locations.
    Returns the train and test data if successful, None otherwise.
    """
    try:
        # Try multiple possible paths for the data
        data_paths = [
            'train_data.csv',
            '../data/train_data.csv',
            'data/train_data.csv',
            os.path.join(os.path.dirname(__file__), 'train_data.csv'),
            os.path.join(os.path.dirname(__file__), '../data/train_data.csv'),
            os.path.join(os.path.dirname(__file__), 'data/train_data.csv')
        ]
        
        train_data = None
        for path in data_paths:
            try:
                st.write(f"Attempting to load train data from: {os.path.abspath(path)}")
                train_data = pd.read_csv(path, encoding='ISO-8859-1')
                st.write(f"Train data loaded successfully from {path}")
                break
            except Exception as e:
                st.write(f"Could not load train data from {path}: {str(e)}")
                continue
        
        if train_data is None:
            raise FileNotFoundError("Could not find the train data file in any of the expected locations")
        
        # Try multiple possible paths for the test data
        test_paths = [
            'test_data.csv',
            '../data/test_data.csv',
            'data/test_data.csv',
            os.path.join(os.path.dirname(__file__), 'test_data.csv'),
            os.path.join(os.path.dirname(__file__), '../data/test_data.csv'),
            os.path.join(os.path.dirname(__file__), 'data/test_data.csv')
        ]
        
        test_data = None
        for path in test_paths:
            try:
                st.write(f"Attempting to load test data from: {os.path.abspath(path)}")
                test_data = pd.read_csv(path, encoding='ISO-8859-1')
                st.write(f"Test data loaded successfully from {path}")
                break
            except Exception as e:
                st.write(f"Could not load test data from {path}: {str(e)}")
                continue
        
        if test_data is None:
            st.warning("Could not find the test data file. Proceeding with only train data.")
        
        return train_data, test_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Main title
st.title("🔧 Material Hardness Prediction")
st.markdown("---")

# Load model and data
model, selected_features = load_model()
train_data, test_data = load_data()

if model is None or train_data is None:
    st.error("Failed to load model or data. Please check the error messages above.")
else:
    st.write("Model type:", type(model).__name__)
    st.write("Selected features:", selected_features)
    st.write("Train data shape:", train_data.shape)
    st.write("Test data shape:", test_data.shape)

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select a Page", ["Home", "Make Prediction", "Model Performance", "Data Analysis"])

    if page == "Home":
        st.header("Welcome to the Hardness Prediction System")
        st.write("""
        This application helps predict material hardness based on various input parameters. 
        The model uses machine learning to make accurate predictions based on historical data.
        """)
        
        # Display debug information
        st.subheader("Debug Information")
        st.write(f"Model type: {type(model).__name__}")
        st.write(f"Number of selected features: {len(selected_features)}")
        st.write("Selected features:", selected_features)
        
        # Display some information about the dataset
        st.subheader("Dataset Information")
        st.write(f"Number of training samples: {train_data.shape[0]}")
        st.write(f"Number of features in training data: {train_data.shape[1]}")
        st.write("First few rows of the training data:")
        st.write(train_data.head())
        
    elif page == "Make Prediction":
        st.header("Make a Prediction")
        st.write("Enter the values for prediction:")
        
        # Create input fields for each feature
        input_data = {}
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(selected_features):
            with col1 if i % 2 == 0 else col2:
                input_data[feature] = st.number_input(
                    f"{feature}",
                    value=float(train_data[feature].mean()),
                    format="%.4f"
                )
        
        if st.button("Predict"):
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(input_df)
            
            st.success(f"Predicted Hardness: {prediction[0]:.2f}")
            
    elif page == "Model Performance":
        st.header("Model Performance Metrics")
        
        # Check if 'Hardness (HVN)' column exists in test_data
        if 'Hardness (HVN)' in test_data.columns:
            # Make predictions on test data
            X_test = test_data[selected_features]
            y_test = test_data['Hardness (HVN)']
            
            # Use the evaluate_model function from model_evaluation.py
            y_pred, metrics = evaluate_model(model, X_test, y_test)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Squared Error", f"{metrics['MSE']:.4f}")
            with col2:
                st.metric("Root MSE", f"{metrics['RMSE']:.4f}")
            with col3:
                st.metric("Mean Absolute Error", f"{metrics['MAE']:.4f}")
            with col4:
                st.metric("R² Score", f"{metrics['R2']:.4f}")
            
            # Create tabs for different plots
            tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Residuals", "Error Distribution"])
            
            with tab1:
                # Plot actual vs predicted using the function from model_evaluation.py
                fig = plot_predictions(y_test, y_pred)
                st.pyplot(fig)
            
            with tab2:
                # Plot residuals using the function from model_evaluation.py
                fig = plot_residuals(y_test, y_pred)
                st.pyplot(fig)
            
            with tab3:
                # Plot error distribution using the function from model_evaluation.py
                fig = plot_error_distribution(y_test, y_pred)
                st.pyplot(fig)
        else:
            st.error("The test data does not contain a 'Hardness (HVN)' column. Available columns are: " + ", ".join(test_data.columns))
        
    else:  # Data Analysis
        st.header("Data Analysis")

        # Data overview
        st.subheader("Data Overview")
        st.write(train_data.describe())

        # Feature distributions
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select Feature", selected_features)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=train_data, x=feature, bins=30)
        plt.title(f"Distribution of {feature}")
        st.pyplot(fig)

        # Feature vs Target (check if 'Hardness' or 'Hardness (HVN)' exists)
        hardness_col = None
        if 'Hardness' in train_data.columns:
            hardness_col = 'Hardness'
        elif 'Hardness (HVN)' in train_data.columns:
            hardness_col = 'Hardness (HVN)'

        if hardness_col:
            st.subheader("Feature vs Target")
            feature_vs_target = st.selectbox("Select Feature to Plot Against Hardness", selected_features, key="feature_vs_target")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=train_data, x=feature_vs_target, y=hardness_col)
            plt.title(f"{feature_vs_target} vs {hardness_col}")
            st.pyplot(fig)
        else:
            st.error("No hardness column found in the training data.")
