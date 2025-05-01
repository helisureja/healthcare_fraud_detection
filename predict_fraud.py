import pandas as pd
import pickle

# Load the trained model
with open("fraud_detection_model.pkl", "rb") as f:
    model = pickle.load(f)

# Function to predict fraud based on user input
def predict_fraud(input_df):
    # Assume 'TotalClaimAmount' is the required feature
    prediction = model.predict(input_df)
    input_df['Prediction'] = prediction
    input_df['Prediction'] = input_df['Prediction'].map({1: 'Fraud', 0: 'Not Fraud'})
    return input_df

# Example usage (for testing outside Power BI)
if __name__ == "__main__":
    test_data = pd.DataFrame({
        'TotalClaimAmount': [5000],  # Add more features if your model uses more
    })
    result = predict_fraud(test_data)
    print(result)
