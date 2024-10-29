import pandas as pd
import joblib

# Step 1: Load the trained model and scaler
model_filename = "optimized_random_forest_model.pkl"
scaler_filename = "scaler.pkl"
model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Step 2: Create a sample data frame with the same structure as the training data
sample_data = {
    'credit_score': [300, 400, 350],         # Lower credit scores are associated with higher churn
    'country': [2, 0, 1],                    # Assuming these country encodings match the high churn countries
    'gender': [1, 1, 0],                     # Gender may have some effect, keeping it consistent
    'age': [65, 70, 75],                     # Older customers tend to churn more
    'tenure': [1, 2, 1],                     # Low tenure means recent customers are more likely to churn
    'balance': [0.00, 0.00, 0.00],           # Zero balance indicates no engagement with the bank
    'products_number': [4, 3, 4],             # High number of products can sometimes lead to dissatisfaction
    'credit_card': [0, 0, 0],                 # No credit card, indicating lack of commitment to the bank
    'active_member': [0, 0, 0],               # Not an active member, making churn likely
    'estimated_salary': [50000.00, 30000.00, 25000.00]  # Low to moderate salaries
}


# Convert the sample data into a DataFrame
sample_df = pd.DataFrame(sample_data)

# Step 3: Standardize the features using the same scaler from training
sample_df_scaled = scaler.transform(sample_df)

# Step 4: Use the loaded model to make predictions
predictions = model.predict(sample_df_scaled)

# Step 5: Display the predictions
print("Sample Data:")
print(sample_df)
print("\nPredicted Churn (1 = Churn, 0 = No Churn):")
print(predictions)
