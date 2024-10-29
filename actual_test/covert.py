import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load the scaler and the trained model
scaler = joblib.load('scaler.pkl')
model = tf.keras.models.load_model('trained_neural_network_model.h5')

# Load the churn_converted.csv dataset
data = pd.read_csv('churn_converted.csv')

# Select relevant columns and add dummy values for the remaining ones
X = data[['account_length', 'total_day_minutes', 'total_intl_charge']]

# Add dummy values for the missing features
X['dummy_credit_score'] = 600  # Dummy credit score
X['dummy_estimated_salary'] = 50000  # Dummy estimated salary

# Reorder the columns to match the model's input
final_data = X[['dummy_credit_score', 'account_length', 'total_day_minutes', 
                'dummy_estimated_salary', 'total_intl_charge']]

# Use 'class' as the target column
y_true = data['class']

# Convert X to a numpy array
X_np = final_data.to_numpy()

# Apply the scaler to the 5 relevant columns that were scaled during model training
X_np = scaler.transform(X_np)

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
