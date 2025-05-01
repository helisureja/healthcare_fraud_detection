import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
def load_data():
    train = pd.read_csv('Train-1542865627584.csv')
    train_beneficiary = pd.read_csv('Train_Beneficiarydata-1542865627584.csv')
    train_inpatient = pd.read_csv('Train_Inpatientdata-1542865627584.csv')
    train_outpatient = pd.read_csv('Train_Outpatientdata-1542865627584.csv')
    return train, train_beneficiary, train_inpatient, train_outpatient

# 2. Preprocess Data
def preprocess_data(train, beneficiary, inpatient, outpatient):
    # Merge inpatient and outpatient data
    claims = pd.concat([inpatient, outpatient], ignore_index=True)
    
    # Merge with beneficiary data
    claims = claims.merge(beneficiary, on='BeneID', how='left')
    
    # Merge with train data to get 'PotentialFraud' label
    claims = claims.merge(train[['Provider', 'PotentialFraud']], on='Provider', how='left')
    
    # Feature Engineering: Example - Calculate total claim amount
    claims['TotalClaimAmount'] = claims['InscClaimAmtReimbursed']
    
    # Handle missing values
    claims.fillna(0, inplace=True)
    
    # Encode categorical variables
    claims['PotentialFraud'] = claims['PotentialFraud'].map({'Yes': 1, 'No': 0})
    
    return claims

# 3. Exploratory Data Analysis
def perform_eda(data):
    # Distribution of fraud vs non-fraud
    sns.countplot(x='PotentialFraud', data=data)
    plt.title('Distribution of Potential Fraud')
    plt.savefig('fraud_distribution.png')
    plt.clf()

    # Boxplot of TotalClaimAmount by fraud
    sns.boxplot(x='PotentialFraud', y='TotalClaimAmount', data=data)
    plt.title('Total Claim Amount by Fraud Status')
    plt.savefig('claim_amount_boxplot.png')
    plt.clf()

    # Total claims submitted by each provider (top 20)
    top_providers = data['Provider'].value_counts().nlargest(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_providers.values, y=top_providers.index)
    plt.title('Top 20 Providers by Number of Claims')
    plt.xlabel('Number of Claims')
    plt.savefig('top_providers.png')
    plt.clf()

    # Age distribution by fraud status
    if 'DOB' in data.columns and 'ClaimStartDt' in data.columns:
        data['DOB'] = pd.to_datetime(data['DOB'], errors='coerce')
        data['ClaimStartDt'] = pd.to_datetime(data['ClaimStartDt'], errors='coerce')
        data['Age'] = (data['ClaimStartDt'] - data['DOB']).dt.days // 365
    elif 'Age' not in data.columns:
        print("DOB or ClaimStartDt not found, skipping age plot.")
        return

    sns.histplot(data=data, x='Age', hue='PotentialFraud', bins=30, kde=True)
    plt.title('Age Distribution by Fraud Status')
    plt.savefig('age_distribution.png')
    plt.clf()

    # Average claim amount by chronic condition
    chronic_cols = [col for col in data.columns if 'ChronicCond_' in col]
    for col in chronic_cols:
        plt.figure()
        sns.boxplot(x=data[col], y=data['TotalClaimAmount'], hue=data['PotentialFraud'])
        plt.title(f'Claim Amount vs {col} by Fraud Status')
        plt.savefig(f'{col}_fraud_claim_boxplot.png')
        plt.clf()
#  heatmap of correlations to find strongly linked features
    plt.figure(figsize=(12, 10))
    corr = data.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.clf()

# 4. Train Model
def train_model(data):
    features = ['TotalClaimAmount']  # Add more relevant features as needed
    X = data[features]
    y = data['PotentialFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    with open('fraud_detection_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, X_test, y_test

# 5. Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC Score: {auc}")

# Main Execution
if __name__ == "__main__":
    print("Loading data...")
    train, beneficiary, inpatient, outpatient = load_data()
    
    print("Preprocessing data...")
    data = preprocess_data(train, beneficiary, inpatient, outpatient)
    
    print("Performing Exploratory Data Analysis...")
    perform_eda(data)
    
    print("Training model...")
    model, X_test, y_test = train_model(data)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    print("Analysis complete. Plots saved as 'fraud_distribution.png' and 'claim_amount_boxplot.png'.")
    print("Trained model saved as 'fraud_detection_model.pkl'.")
