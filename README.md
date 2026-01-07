# Telco Customer Churn Prediction

**Project Type:** Machine Learning | Classification  
**Tech Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
**Status:** Completed  

---

## Project Overview

Customer churn is when a customer stops using a company's services. For businesses, predicting churn is critical to take proactive steps to retain valuable customers.  

This project aims to **predict whether a customer will churn** using Telco customer data. It involves data cleaning, exploratory data analysis (EDA), feature engineering, and building machine learning models to predict churn with high accuracy.

---

## Dataset

The dataset used in this project contains customer information from a telecommunications company, including:

- Customer demographics (gender, age, tenure, etc.)
- Account information (contract type, payment method, monthly charges, etc.)
- Services subscribed (internet service, phone service, etc.)
- Churn label (`Yes`/`No`)

**Source:** Public Telco Churn dataset (can include link if applicable, e.g., Kaggle).

---

## Key Features

- `gender`: Male / Female  
- `SeniorCitizen`: 0 or 1  
- `Partner`: Yes / No  
- `Dependents`: Yes / No  
- `tenure`: Number of months with the company  
- `PhoneService`: Yes / No  
- `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`  
- `Contract`: Month-to-month, One year, Two year  
- `PaperlessBilling`: Yes / No  
- `PaymentMethod`: Electronic check, Mailed check, Bank transfer, Credit card  
- `MonthlyCharges`, `TotalCharges`  
- `Churn`: Target variable (`Yes`/`No`)

---

## Steps Performed

1. **Data Cleaning & Preprocessing**
   - Handled missing values and corrected data types.
   - Encoded categorical variables using One-Hot Encoding and Label Encoding.
   - Scaled numeric features where necessary.

2. **Exploratory Data Analysis (EDA)**
   - Analyzed churn distribution and key features using visualization libraries (Matplotlib, Seaborn).  
   - Identified patterns and correlations affecting customer churn.

3. **Handling Class Imbalance**
   - Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.

4. **Feature Engineering**
   - Created new features to improve model performance.
   - Dropped irrelevant or redundant columns.

5. **Model Building**
   - Implemented multiple classification algorithms:
     - Logistic Regression
     - Random Forest
     - K-Nearest Neighbors
     - Support Vector Machine
   - Hyperparameter tuning using GridSearchCV.

6. **Model Evaluation**
   - Metrics used: Accuracy, Precision, Recall, F1-score, ROC-AUC.
   - Confusion Matrix for detailed performance insights.

7. **Insights**
   - Key factors influencing churn: Contract type, tenure, monthly charges, internet service type.
   - Recommended strategies to reduce churn based on model predictions.

---

## Results

- **Best Model:** Random Forest Classifier  
- **Accuracy:** ~82%  
- **ROC-AUC Score:** ~0.84  
- High recall for churned customers, helping in proactive retention campaigns.

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
