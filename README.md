# Password Strength Estimator

This project provides a command-line tool for estimating password strength using machine learning models. Two approaches are implemented:

- **XGBoost Model**
- **Multi-Layer Perceptron (MLP)**

The estimator computes various features from an input password—such as length, character diversity, entropy, and more—to predict its strength category. The output is one of five strength labels: **very_weak**, **weak**, **average**, **strong**, or **very_strong**.

---

## Overview

The estimator extracts features including:

- **Password Length**
- **Number of Uppercase, Lowercase, Digits, and Special Characters**
- **Shannon Entropy**
- **Character Diversity Score**
- **Longest Repeating/Numeric/Letter Sequence**
- **Flags for Common or Reversed Common Passwords**

These features are then fed into two pre-trained machine learning models:

1. **XGBoost Model:**  
   Utilizes gradient boosting on decision trees for prediction. This model also provides feature importance, aiding interpretability.

2. **MLP Model:**  
   Implements a neural network using Keras/TensorFlow. With non-linear activation functions and one or more hidden layers, it effectively captures complex relationships in the feature set.

---

## Features

- **Robust Performance:**  
  Both models achieve an overall accuracy of around **96%** on the test set.

- **Detailed Evaluation Metrics:**
  - **MLP Model Results:**
    - **Test Accuracy:** 96.16%
    - **Precision/Recall/F1-Score:**
      - Class 0: 1.00 / 1.00 / 1.00 (support: 199,940)
      - Class 1: 0.94 / 0.96 / 0.95 (support: 199,875)
      - Class 2: 0.92 / 0.95 / 0.94 (support: 200,084)
      - Class 3: 0.96 / 0.90 / 0.93 (support: 200,245)
      - Class 4: 1.00 / 1.00 / 1.00 (support: 199,875)
    - **Macro/Weighted Average:** 0.96

  - **XGBoost Model Results:**
    - **Test Accuracy:** 96%
    - **Precision/Recall/F1-Score:**
      - Class 0: 1.00 / 1.00 / 1.00 (support: 200,003)
      - Class 1: 0.95 / 0.95 / 0.95 (support: 200,001)
      - Class 2: 0.92 / 0.95 / 0.94 (support: 200,001)
      - Class 3: 0.94 / 0.91 / 0.93 (support: 200,014)
      - Class 4: 1.00 / 1.00 / 1.00 (support: 200,000)
    - **Macro/Weighted Average:** 0.96

---

## Model Comparison

### Overall Accuracy
- **MLP Model:** 96.16%
- **XGBoost Model:** 96%

Both models yield nearly identical accuracy, demonstrating robust performance on the task of password strength estimation.

### Per-Class Performance
- **Extreme Classes (Very Weak and Very Strong):**  
  Both models achieve perfect scores (precision, recall, and f1-score of 1.00).
  
- **Intermediate Classes (Weak, Average, Strong):**
  - **MLP Model:**  
    Slightly reduced performance for intermediate classes with f1-scores ranging from 0.93 to 0.95.
    
  - **XGBoost Model:**  
    Similar trends with intermediate classes showing f1-scores around 0.93–0.95, with a few misclassifications evident in the confusion matrix.

### Methodological Differences
- **MLP Model:**  
  - Captures complex non-linear relationships through neural network architecture.
  - Requires careful tuning of hyperparameters (e.g., number of layers, neurons, learning rate).
  - Typically less interpretable compared to ensemble methods.

- **XGBoost Model:**  
  - Leverages gradient boosting with decision trees.
  - Provides feature importance metrics, offering better interpretability.
  - Known for robustness and efficiency when working with structured data.

### Summary
- **Performance:** Both models exhibit exceptional and nearly identical accuracy (~96%).
- **Strengths:**  
  - The **MLP model** excels in capturing complex patterns with its deep learning architecture.
  - The **XGBoost model** is slightly more interpretable and offers insights into feature importance.
- **Use Case Consideration:**  
  The decision to use one model over the other may depend on specific application requirements such as the need for interpretability, training efficiency, and the nature of the password datasets.

---

## How It Works

1. **Feature Extraction:**  
   The password input is analyzed to compute a set of features like length, diversity, entropy, and sequence patterns.

2. **Model Prediction:**  
   The extracted features are fed into both the XGBoost and MLP models, each of which predicts a strength category.

3. **Output:**  
   The tool outputs a label indicating the password strength, helping users understand the robustness of their passwords.

---

## Usage

Simply run the command-line tool and input the password you want to evaluate. The tool will process the password, extract the relevant features, and output the strength category based on both models.

---

*Note: This README skips detailed installation instructions, assuming that you have set up the required environment and dependencies (Python 3.7+, virtual environment, etc.) beforehand.*
