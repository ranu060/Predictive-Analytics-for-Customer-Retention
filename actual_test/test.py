import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load the scaler and the trained model
scaler = joblib.load('C:/Users/Raghav/Desktop/Model/scaler.pkl')
model = tf.keras.models.load_model('C:/Users/Raghav/Desktop/Model/trained_neural_network_model.keras')

# Load the Churn_Modelling.csv dataset
data = pd.read_csv('C:/Users/Raghav/Documents/GitHub/Predictive-Analytics-for-Customer-Retention/actual_test/Churn_Modelling.csv')

# Select the relevant columns and rename them to match the expected input
data = data[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
             'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']]

# Separate the target (Exited) from the features
X = data.drop(columns=['Exited'])
y_true = data['Exited']  # Actual churn values

# Encode categorical variables (Geography and Gender)
country_mapping = {'France': 0, 'Spain': 1, 'Germany': 2}
gender_mapping = {'Female': 0, 'Male': 1}
X['Geography'] = X['Geography'].map(country_mapping)
X['Gender'] = X['Gender'].map(gender_mapping)

# Ensure no missing values
X = X.dropna()

# Convert the DataFrame to a NumPy array for prediction
X_np = X.to_numpy()

# Apply the scaler to the relevant columns (credit_score, age, tenure, balance, estimated_salary)
X_np[:, [0, 3, 4, 5, 9]] = scaler.transform(X_np[:, [0, 3, 4, 5, 9]])

# Make predictions (churn probabilities)
predictions = model.predict(X_np)

# Convert probabilities to binary classification (1 for churn, 0 for no churn)
y_pred = (predictions > 0.5).astype(int).flatten()

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

# Output the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
