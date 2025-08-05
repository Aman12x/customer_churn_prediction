# Customer Churn Prediction

A machine learning project that predicts customer churn using ensemble methods and provides model interpretability through LIME (Local Interpretable Model-agnostic Explanations).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Data](#data)
- [Results](#results)
- [Model Interpretability](#model-interpretability)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a customer churn prediction system using machine learning techniques. It analyzes customer behavior patterns and subscription data to predict which customers are likely to churn (cancel their subscription). The project includes data preprocessing, model training with multiple algorithms, performance evaluation, and model interpretability features.

## ✨ Features

- **Multiple ML Models**: Random Forest, AdaBoost, and Gradient Boosting classifiers
- **Data Preprocessing**: Handles missing values and feature selection
- **Class Imbalance Handling**: Uses SMOTE (Synthetic Minority Oversampling Technique)
- **Model Evaluation**: ROC curves, confusion matrices, and classification reports
- **Model Interpretability**: LIME explanations for individual predictions
- **Model Persistence**: Saves trained models for future use
- **Visualization**: ROC curves and LIME explanation plots

## 📁 Project Structure

```
customer_churn/
├── input/
│   └── cust_data.csv           # Customer dataset
├── output/
│   └── models/
│       ├── LIME_reports/       # LIME explanation visualizations
│       │   ├── lime_report_ab.jpg
│       │   ├── lime_report_gb.jpg
│       │   └── lime_report_rf.jpg
│       ├── ROC_curves/         # ROC curve plots
│       │   ├── ROC_Curve_ab.png
│       │   ├── ROC_Curve_gb.png
│       │   └── ROC_Curve_rf.png
│       ├── model_ab.pkl        # Trained AdaBoost model
│       ├── model_gb.pkl        # Trained Gradient Boosting model
│       └── model_rf.pkl        # Trained Random Forest model
├── src/
│   ├── ML_Pipeline/
│   │   ├── evaluate_metrics.py # Model evaluation functions
│   │   ├── lime.py            # LIME explanation implementation
│   │   ├── ml_model.py        # Model training and prediction
│   │   └── utils.py           # Data preprocessing utilities
│   └── main.py                # Main execution script
├── requirements.txt           # Project dependencies
└── README.md                 # Project documentation
```

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd customer_churn
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Complete Pipeline

Execute the main script to run the entire machine learning pipeline:

```bash
cd src
python main.py
```

This will:
- Load and preprocess the customer data
- Train the selected model (Random Forest by default)
- Generate evaluation metrics
- Create ROC curve visualization
- Generate LIME explanations
- Save the trained model

### Using Individual Components

#### Data Preprocessing
```python
from ML_Pipeline.utils import read_data, inspection, null_values

# Load data
df = read_data("path/to/cust_data.csv")

# Inspect data
inspection(df)

# Handle missing values
df_clean = null_values(df)
```

#### Model Training
```python
from ML_Pipeline.ml_model import prepare_model_smote, run_model

# Prepare data with SMOTE
X_train, X_test, y_train, y_test = prepare_model_smote(
    df, 
    class_col="churn", 
    cols_to_exclude=["customer_id", "phone_no", "year"]
)

# Train model
model, predictions = run_model("random", X_train, X_test, y_train, y_test)
```

#### Model Evaluation
```python
from ML_Pipeline.evaluate_metrics import confusion_matrix, roc_curve

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Create ROC curve
roc_val = roc_curve(model, X_test, y_test)
```

## 🤖 Models

The project implements three ensemble learning algorithms:

### 1. Random Forest Classifier
- **Configuration**: `max_depth=5`
- **Use Case**: Default model with good balance of performance and interpretability
- **Strengths**: Handles overfitting well, provides feature importance

### 2. AdaBoost Classifier
- **Configuration**: `n_estimators=100`
- **Use Case**: Sequential ensemble method
- **Strengths**: Good for weak learners, adaptive boosting

### 3. Gradient Boosting Classifier
- **Configuration**: Default parameters
- **Use Case**: Advanced boosting technique
- **Strengths**: Often provides high accuracy, handles various data types

## 📊 Data

### Dataset Features

The customer dataset includes the following features:

- **Customer Information**: `customer_id`, `phone_no`, `gender`, `age`
- **Subscription Details**: `year`, `no_of_days_subscribed`, `multi_screen`, `mail_subscribed`
- **Usage Patterns**: `weekly_mins_watched`, `minimum_daily_mins`, `maximum_daily_mins`, `weekly_max_night_mins`
- **Engagement Metrics**: `videos_watched`, `maximum_days_inactive`, `customer_support_calls`
- **Target Variable**: `churn` (0 = No Churn, 1 = Churn)

### Data Preprocessing Steps

1. **Missing Value Handling**: Removes rows with null values
2. **Feature Selection**: Excludes non-predictive columns (`customer_id`, `phone_no`, `year`)
3. **Numerical Feature Focus**: Selects only numerical columns for modeling
4. **Class Balancing**: Applies SMOTE to handle class imbalance

## 📈 Results

The models generate several evaluation metrics:

- **Classification Report**: Precision, Recall, F1-score for each class
- **ROC-AUC Score**: Area under the ROC curve
- **Confusion Matrix**: True/False Positives and Negatives
- **ROC Curves**: Visual representation of model performance

Example output:
```
The area under the curve is: 0.XX
```

## 🔍 Model Interpretability

### LIME (Local Interpretable Model-agnostic Explanations)

The project uses LIME to explain individual predictions:

- **Purpose**: Understand which features contribute most to a specific prediction
- **Output**: Visual explanations showing feature importance for individual instances
- **Usage**: Helps build trust in model decisions and identify potential biases

### Generated Visualizations

- **ROC Curves**: Compare model performance across different thresholds
- **LIME Reports**: Feature importance for individual predictions
- **Confusion Matrices**: Detailed breakdown of prediction accuracy

## 🛠️ Dependencies

- `numpy==1.19.5`: Numerical computing
- `matplotlib==3.2.2`: Plotting and visualization
- `pandas==1.3.0`: Data manipulation and analysis
- `imblearn==0.0`: Imbalanced dataset handling (SMOTE)
- `scikit_learn==1.0.2`: Machine learning algorithms
- `lime==0.2.0.1`: Model interpretability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 Notes

- The project is configured to run Random Forest by default. To use other models, modify the `model` parameter in `main.py`:
  - `"random"` for Random Forest
  - `"adaboost"` for AdaBoost
  - `"gradient"` for Gradient Boosting

- All output files (models, plots, reports) are saved in the `output/` directory

- The LIME explanations are generated for a single instance (index 1) by default. Modify the `chosen_index` parameter to explain different predictions.

## 🔧 Troubleshooting

- **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
- **Path Issues**: Run the script from the `src/` directory
- **Memory Issues**: For large datasets, consider reducing the dataset size or using more efficient algorithms
- **Missing Output Directories**: The script assumes output directories exist; create them manually if needed

---

*This project demonstrates end-to-end machine learning pipeline implementation with a focus on model interpretability and practical deployment considerations.*