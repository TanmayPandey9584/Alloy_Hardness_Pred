# Hardness Prediction Model

This project implements a machine learning model for predicting material hardness based on various input parameters.

## Project Structure

```
HardnessPred/
├── data/               # Data files
│   ├── raw/           # Original dataset
│   └── processed/     # Processed datasets
├── models/            # Saved model files
├── src/              # Source code
│   ├── training.py   # Model training script
│   └── evaluation.py # Model evaluation script
├── results/          # Output files and visualizations
└── README.md         # Project documentation
```

## Files Description

### Data Files
- `data.csv`: Original dataset
- `train_data.csv`: Training dataset
- `test_data.csv`: Testing dataset

### Source Code
- `Updated_Model.py`: Main model implementation with feature selection and hyperparameter tuning
- `model_evaluation.py`: Script for model evaluation and performance visualization

### Model Files
- `Hardness_Prediction_Model.pkl`: Trained model
- `Selected_Features.json`: Selected features configuration

## Usage

1. Data Preparation:
   - Place your input data in the `data/` directory
   - Run data preprocessing scripts if needed

2. Model Training:
   ```bash
   python src/training.py
   ```

3. Model Evaluation:
   ```bash
   python src/evaluation.py
   ```

## Results

The model's performance is evaluated using various metrics:
- Mean Squared Error (MSE)
- R-squared Score
- Feature Importance Analysis
- Error Distribution Analysis

Results and visualizations are saved in the `results/` directory. 