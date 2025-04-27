import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import pickle

def load_model_and_features():
    try:
        # Try different possible paths for the model file
        model_paths = [
            "Hardness_Prediction_Model.pkl",
            "../models/Hardness_Prediction_Model.pkl",
            "models/Hardness_Prediction_Model.pkl"
        ]
        
        model = None
        for path in model_paths:
            try:
                with open(path, 'rb') as file:
                    model = pickle.load(file)
                print(f"Model loaded successfully from {path}")
                break
            except FileNotFoundError:
                continue
        
        if model is None:
            raise FileNotFoundError("Could not find the model file in any of the expected locations")
        
        # Try different possible paths for the features file
        feature_paths = [
            "Selected_Features.json",
            "../models/Selected_Features.json",
            "models/Selected_Features.json",
            "Selected_Fearures.json",  # Handle the typo in the original filename
            "../models/Selected_Fearures.json",
            "models/Selected_Fearures.json"
        ]
        
        selected_features = None
        for path in feature_paths:
            try:
                with open(path, 'r') as f:
                    selected_features = json.load(f)
                print(f"Features loaded successfully from {path}")
                break
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        
        if selected_features is None:
            raise FileNotFoundError("Could not find the features file in any of the expected locations")
        
        return model, selected_features
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        raise
    except Exception as e:
        print(f"Error loading model or features: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return predictions and metrics"""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        return y_pred, metrics
    except Exception as e:
        print(f"Error evaluating model: {e}")
        raise

def plot_predictions(y_test, y_pred, ax=None):
    """Plot actual vs predicted values"""
    try:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use seaborn for better aesthetics
        sns.set_style("whitegrid")
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
        
        # Add perfect prediction line
        line_x = np.linspace(min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()), 100)
        ax.plot(line_x, line_x, 'r--', lw=2)
        
        ax.set_xlabel("Actual Hardness")
        ax.set_ylabel("Predicted Hardness")
        ax.set_title("Actual vs Predicted Hardness")
        
        return ax.figure
    except Exception as e:
        print(f"Error creating prediction plot: {e}")
        return None

def plot_residuals(y_test, y_pred, ax=None):
    """Plot residuals vs predicted values"""
    try:
        residuals = y_test - y_pred
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use seaborn for better aesthetics
        sns.set_style("whitegrid")
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=ax)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        ax.set_xlabel("Predicted Hardness")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        
        return ax.figure
    except Exception as e:
        print(f"Error creating residual plot: {e}")
        return None

def plot_error_distribution(y_test, y_pred, ax=None):
    """Plot distribution of prediction errors"""
    try:
        errors = y_pred - y_test
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use seaborn for better aesthetics
        sns.set_style("whitegrid")
        sns.histplot(errors, bins=30, kde=True, ax=ax)
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Prediction Errors")
        
        return ax.figure
    except Exception as e:
        print(f"Error creating error distribution plot: {e}")
        return None

def main():
    """Main function to run model evaluation"""
    try:
        # Set seaborn style for all plots
        sns.set_theme(style="whitegrid")
        
        # Load model and features
        print("Loading model and features...")
        model, selected_features = load_model_and_features()
        print(f"Model type: {type(model).__name__}")
        print(f"Selected features: {selected_features}")
        
        # Load test data
        test_path = '../data/test_data.csv'
        print(f"Loading test data from: {os.path.abspath(test_path)}")
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data file not found at: {os.path.abspath(test_path)}")
            
        test_data = pd.read_csv(test_path)
        print(f"Test data shape: {test_data.shape}")
        print(f"Test data columns: {test_data.columns.tolist()}")
        
        # Verify all selected features are in the test data
        missing_features = [f for f in selected_features if f not in test_data.columns]
        if missing_features:
            raise ValueError(f"The following features are missing from the test data: {missing_features}")
        
        # Prepare test data - Fix: Use "Hardness (HVN)" instead of "Hardness"
        X_test = test_data[selected_features]
        
        # Check if the hardness column exists with the correct name
        if 'Hardness (HVN)' not in test_data.columns:
            raise ValueError("The 'Hardness (HVN)' column is missing from the test data")
            
        y_test = test_data['Hardness (HVN)']
        
        # Evaluate model
        print("Evaluating model...")
        y_pred, metrics = evaluate_model(model, X_test, y_test)
        
        # Print metrics
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Create plots
        print("Creating plots...")
        fig1 = plot_predictions(y_test, y_pred)
        fig2 = plot_residuals(y_test, y_pred)
        fig3 = plot_error_distribution(y_test, y_pred)
        
        plt.show()
        print("Evaluation complete!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()