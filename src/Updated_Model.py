# =============================================================================
# 0. importing libraries
# =============================================================================

import pandas as pd
import json
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.feature_selection import RFECV

# =============================================================================
# 1. Data Loading and Preparation
# =============================================================================

# Define paths - Fixed the path to include 'Projects' folder
data_dir = "D:\\Python\\Pycharm3.9\\Projects\\HardnessPred\\data"
models_dir = "D:\\Python\\Pycharm3.9\\Projects\\HardnessPred\\models"

# Create models directory if it doesn't exist
os.makedirs(models_dir, exist_ok=True)

# Print current working directory and check if file exists
print(f"Current working directory: {os.getcwd()}")
print(f"Checking if data file exists: {os.path.exists(os.path.join(data_dir, 'train_data.csv'))}")

# Load the training data file
train_data_path = os.path.join(data_dir, "train_data.csv")
train_data = pd.read_csv(train_data_path, encoding='ISO-8859-1')

# Separate features and target variable
x = train_data.drop("Hardness (HVN)", axis=1)
y = train_data["Hardness (HVN)"]

# Split data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# =============================================================================
# 2. Apply RFECV for Feature Selection
# =============================================================================

# Initialize the base estimator (using XGBRegressor)
base_estimator = XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')

# Use RFECV to determine the optimal number of features
# using 5-fold CV and R2 as the scoring metric
rfecv = RFECV(base_estimator, step=1, cv=5, scoring='r2')
rfecv.fit(x_train, y_train)
x_train_selected = x_train[x_train.columns[rfecv.support_]]

# Get the selected features
selected_features = list(x_train.columns[rfecv.support_])
features_path = os.path.join(models_dir, "Selected_Features.json")  # Fixed typo in filename
with open(features_path, "w") as f:
    json.dump(selected_features, f)
print(f"Selected features saved to: {features_path}")

x_test_selected = x_test[selected_features]

# =============================================================================
# 3. Define Hyperparameter Grid for XGBRegressor
# =============================================================================

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [1, 3, 5],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1],
    'colsample_bytree': [0.8, 0.9, 1],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}

# =============================================================================
# 4. Initialize the XGBRegressor and GridSearchCV
# =============================================================================

# Setup GridSearchCV with 5-fold cross-validation
model = GridSearchCV(base_estimator, param_grid, cv=5, n_jobs=-1, scoring="r2")

# Calculate the total iterations (number of parameter combinations * number of folds)
total_iterations = len(param_grid['learning_rate']) * len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_child_weight']) * len(param_grid['gamma']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree']) * len(param_grid['reg_alpha']) * len(param_grid['reg_lambda']) * 5

# =============================================================================
# 5. Run Grid Search with a Progress Bar
# =============================================================================

with tqdm_joblib(tqdm(desc="GridSearchCV", total=total_iterations)):
    model.fit(x_train_selected, y_train)

print(f"Best Params are: {model.best_params_}")

# =============================================================================
# 6. Evaluate the model on test data
# =============================================================================

# Get the best model from GridSearchCV
best_model = model.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(x_test_selected)

# Calculate and print evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Mean Squared Error: {mse:.4f}")
print(f"Test R² Score: {r2:.4f}")

# =============================================================================
# 7. Save the Final Model
# =============================================================================

model_path = os.path.join(models_dir, "Hardness_Prediction_Model.pkl")
with open(model_path, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved successfully to: {model_path}")

# Save test data for later evaluation
test_data = pd.concat([x_test, y_test], axis=1)
test_data_path = os.path.join(data_dir, "test_data.csv")
test_data.to_csv(test_data_path, index=False)
print(f"Test data saved to: {test_data_path}")