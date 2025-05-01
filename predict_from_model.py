import pandas as pd
import pickle

# Load the model (make sure the path is correct on your PC!)
model = pickle.load(open(r"C:\Users\Admin\Desktop\healthcare_fraud_detection\healthcare_fraud_detection\fraud_detection_model.pkl", "rb"))

# Power BI provides 'dataset' as input
input_df = dataset

# Select only required features (adjust this based on your model)
input_df['Prediction'] = model.predict(input_df[['TotalClaimAmount']])
input_df['Prediction'] = input_df['Prediction'].map({1: 'Fraud', 0: 'Not Fraud'})

result = input_df
