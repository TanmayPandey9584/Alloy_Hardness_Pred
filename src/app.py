import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import os
from model_evaluation import (
    load_model_and_features, 
    evaluate_model, 
    plot_predictions, 
    plot_residuals, 
    plot_error_distribution
)

# Set page configuration
st.set_page_config(
    page_title="Hardness Prediction Model",
    page_icon="🔧",
    layout="wide"
)

# Load the model and selected features
def load_model():
    try:
        model_path = '../models/Hardness_Prediction_Model.pkl'
        features_path = '../models/Selected_Features.json'
        
        st.write(f"Attempting to load model ")
        st.write(f"Attempting to load features")
        
        model, selected_features = load_model_and_features(model_path, features_path)
        
        st.write("Model and features loaded successfully")
        return model, selected_features
    except Exception as e:
        st.error(f"Error loading model or features: {str(e)}")
        return None, None

# Load the data
def load_data():
    try:
        train_path = '../data/train_data.csv'
        test_path = '../data/test_data.csv'
        
        #st.write(f"Attempting to load train data from: {os.path.abspath(train_path)}")
        train_data = pd.read_csv(train_path)
        st.write("Train data loaded successfully")
        
        #st.write(f"Attempting to load test data from: {os.path.abspath(test_path)}")
        test_data = pd.read_csv(test_path)
        st.write("Test data loaded successfully")
        
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