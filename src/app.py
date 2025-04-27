import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

# Configure the page
st.set_page_config(
    page_title="Alloy Hardness Prediction",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Function to load the model and selected features
def load_model(verbose=False):
    """
    Load the trained model and selected features.
    If verbose is True, show detailed loading information.
    """
    try:
        # List of possible locations for the model file
        model_paths = [
            'models/Hardness_Prediction_Model.pkl',
            'Hardness_Prediction_Model.pkl',
            '../models/Hardness_Prediction_Model.pkl',
        ]

        # Try each path until we find the model
        model = None
        model_path_used = None
        for path in model_paths:
            try:
                if verbose:
                    st.write(f"Trying to load model from: {os.path.abspath(path)}")
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

        # If we couldn't find the model, raise an error
        if model is None:
            raise FileNotFoundError("Could not find the model file")

        # List of possible locations for the features file
        feature_paths = [
            'models/Selected_Features.json',
            'models/Selected_Fearures.json',  # Handle the typo in the original filename
            'Selected_Features.json',
            'Selected_Fearures.json',
            '../models/Selected_Features.json',
            '../models/Selected_Fearures.json',
        ]

        # Try each path until we find the features
        selected_features = None
        feature_path_used = None
        for path in feature_paths:
            try:
                if verbose:
                    st.write(f"Trying to load features from: {os.path.abspath(path)}")
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

        # If we couldn't find the features, raise an error
        if selected_features is None:
            raise FileNotFoundError("Could not find the features file")

        # Show success messages if not in verbose mode
        if not verbose:
            st.success(f"✅ Model loaded from {model_path_used}")
            st.success(f"✅ Features loaded from {feature_path_used}")

        return model, selected_features
    except Exception as e:
        st.error(f"Error loading model or features: {str(e)}")
        return None, None


# Function to load the training data
def load_data(verbose=False):
    """
    Load the training data.
    If verbose is True, show detailed loading information.
    """
    try:
        # List of possible locations for the data file
        data_paths = [
            'data/train_data.csv',
            'train_data.csv',
            '../data/train_data.csv',
        ]

        # Try each path until we find the data
        train_data = None
        train_path_used = None
        for path in data_paths:
            try:
                if verbose:
                    st.write(f"Trying to load train data from: {os.path.abspath(path)}")
                train_data = pd.read_csv(path, encoding='ISO-8859-1')
                train_path_used = path
                if verbose:
                    st.write(f"Train data loaded successfully from {path}")
                break
            except Exception as e:
                if verbose:
                    st.write(f"Could not load train data from {path}: {str(e)}")
                continue

        # If we couldn't find the data, raise an error
        if train_data is None:
            raise FileNotFoundError("Could not find the train data file")

        # Show success message if not in verbose mode
        if not verbose:
            st.success(f"✅ Training data loaded from {train_path_used}")

        return train_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Make Prediction", "Data Exploration", "About"])

# Add a debug mode toggle in the sidebar
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# HOME PAGE
if page == "Home":
    # Display the main header
    st.title("Alloy Hardness Prediction System")

    # Display welcome message
    st.info(
        "Welcome to the Alloy Hardness Prediction System. This application uses machine learning to predict the hardness of alloys based on their composition and processing parameters.")

    # Display instructions
    st.header("How to use this application:")
    st.write(
        "1. **Make Prediction**: Enter the composition and processing parameters of your alloy to get a hardness prediction.")
    st.write(
        "2. **Data Exploration**: Explore the training data and understand the relationships between different features.")
    st.write("3. **About**: Learn more about the model and the science behind hardness prediction.")

    # Load model and data in the background
    with st.spinner("Loading model and data..."):
        model, selected_features = load_model(verbose=debug_mode)
        train_data = load_data(verbose=debug_mode)

    # Show status message
    if model is not None and train_data is not None:
        st.success("✅ System is ready for predictions!")
    else:
        st.error("❌ Failed to load model or data. Please check the error messages above.")

# PREDICTION PAGE
elif page == "Make Prediction":
    # Display the main header
    st.title("Make a Hardness Prediction")

    # Load model and features
    model, selected_features = load_model(verbose=debug_mode)

    # Check if model and features loaded successfully
    if model is None or selected_features is None:
        st.error("❌ Failed to load model or features. Please check the error messages above.")
    else:
        # Display instructions
        st.info("Enter the values for each feature to get a hardness prediction.")

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
            st.subheader("Prediction Result")
            st.success(f"Predicted Hardness (HVN): {prediction[0]:.2f}")

# DATA EXPLORATION PAGE
elif page == "Data Exploration":
    # Display the main header
    st.title("Data Exploration")

    # Load data
    train_data = load_data(verbose=debug_mode)

    # Check if data loaded successfully
    if train_data is None:
        st.error("❌ Failed to load data. Please check the error messages above.")
    else:
        # Display instructions
        st.info("Explore the training data to understand the relationships between different features.")

        # Display data overview
        st.header("Data Overview")
        st.write(f"Number of samples: {train_data.shape[0]}")
        st.write(f"Number of features: {train_data.shape[1] - 1}")  # Excluding the target variable

        # Display first few rows of the data
        st.header("Sample Data")
        st.dataframe(train_data.head())

        # Display statistics
        st.header("Statistical Summary")
        st.dataframe(train_data.describe())

        # Correlation heatmap
        st.header("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = train_data.corr()
        sns.heatmap(correlation, annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Feature vs Target plots
        st.header("Feature vs Hardness")

        # Let user select a feature to plot against hardness
        feature_to_plot = st.selectbox("Select a feature to plot against Hardness:",
                                       train_data.columns.drop("Hardness (HVN)"))

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=feature_to_plot, y="Hardness (HVN)", data=train_data, ax=ax)
        ax.set_title(f"{feature_to_plot} vs Hardness")
        st.pyplot(fig)

elif page == "About":
    st.title("About the Hardness Prediction Model")

    st.info(
        "This application uses a machine learning model to predict the hardness of alloys based on their composition and processing parameters.")

    st.header("Model Information:")
    st.write("- **Model Type**: XGBoost Regressor")
    st.write("- **Feature Selection**: Recursive Feature Elimination with Cross-Validation (RFECV)")
    st.write("- **Hyperparameter Tuning**: Grid Search with Cross-Validation")

    st.header("About Hardness Prediction:")
    st.write(
        "Hardness is a measure of the resistance of a material to localized plastic deformation. It is an important property for many engineering applications, as it can indicate wear resistance, strength, and other mechanical properties.")
    st.write(
        "Predicting hardness based on composition and processing parameters can help in the design of new alloys with desired properties, reducing the need for extensive experimental testing.")

    st.header("References:")
    st.write("1. XGBoost: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)")
    st.write("2. Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)")
    st.write("3. Streamlit: [https://streamlit.io/](https://streamlit.io/)")
