import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load the scaler and the trained model
scaler = joblib.load('scaler.pkl')
model = tf.keras.models.load_model('trained_neural_network_model.h5')

# Load the botswana_bank_customer_churn.csv dataset
data = pd.read_csv('botswana_bank_customer_churn.csv')

# Select relevant columns for the model
X = data[['Credit Score', 'Customer Tenure', 'Balance', 'NumOfProducts', 'Outstanding Loans']]

# Add dummy values for the missing features
X['dummy_credit_card'] = 1  # Dummy value for credit card
X['dummy_active_member'] = 1  # Dummy active member
X['dummy_estimated_salary'] = 50000  # Dummy estimated salary
X['dummy_feature1'] = 0  # Additional dummy feature to match the model's input
X['dummy_feature2'] = 0  # Another additional dummy feature

# Reorder the columns to match the model's input expectations
final_data = X[['Credit Score', 'Customer Tenure', 'Balance', 
                'NumOfProducts', 'Outstanding Loans', 'dummy_credit_card', 
                'dummy_active_member', 'dummy_estimated_salary', 
                'dummy_feature1', 'dummy_feature2']]

# Use 'Churn Flag' as the target column
y_true = data['Churn Flag']

# Convert X to a numpy array
X_np = final_data.to_numpy()

# Apply the scaler to the relevant columns
X_np[:, [0, 1, 2, 4, 7]] = scaler.transform(X_np[:, [0, 1, 2, 4, 7]])

# Make predictions using the loaded model
predictions = model.predict(X_np)

# Convert probabilities to binary classification (1 for churn, 0 for no churn)
y_pred = (predictions > 0.5).astype(int).flatten()

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

# Output the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
