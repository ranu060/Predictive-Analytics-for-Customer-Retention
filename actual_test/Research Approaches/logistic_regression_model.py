# LOGISTIC REGRESSION APPROACH


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the datasets
dataset1 = pd.read_csv("C:/Users/Raghav/Documents/GitHub/Predictive-Analytics-for-Customer-Retention/actual_test/Churn_Modelling.csv")
dataset2 = pd.read_csv("C:/Users/Raghav/Documents/GitHub/Predictive-Analytics-for-Customer-Retention/actual_test/Bank Customer Churn Prediction.csv")
dataset3 = pd.read_csv("C:/Users/Raghav/Documents/GitHub/Predictive-Analytics-for-Customer-Retention/actual_test/Customer-Churn-Records.csv")

# Step 2: Standardize column names and align features for dataset1 and dataset3
rename_map = {
    'CustomerId': 'customer_id',
    'Geography': 'country',
    'NumOfProducts': 'products_number',
    'HasCrCard': 'credit_card',
    'IsActiveMember': 'active_member',
    'EstimatedSalary': 'estimated_salary',
    'Exited': 'churn',
    'CreditScore': 'credit_score',
    'Gender': 'gender',
    'Age': 'age',
    'Tenure': 'tenure',
    'Balance': 'balance'
}
dataset1 = dataset1.rename(columns=rename_map)
dataset3 = dataset3.rename(columns=rename_map)

# Step 3: Select common columns
columns_to_use = [
    'customer_id', 'credit_score', 'country', 'gender', 'age', 'tenure', 'balance',
    'products_number', 'credit_card', 'active_member', 'estimated_salary', 'churn'
]
dataset1_filtered = dataset1[columns_to_use]
dataset2_filtered = dataset2[columns_to_use]
dataset3_filtered = dataset3[columns_to_use]

# Step 4: Combine the datasets
combined_data = pd.concat([dataset1_filtered, dataset2_filtered, dataset3_filtered], ignore_index=True)

# Step 5: Encode categorical variables
le_country = LabelEncoder()
le_gender = LabelEncoder()
combined_data['country'] = le_country.fit_transform(combined_data['country'])
combined_data['gender'] = le_gender.fit_transform(combined_data['gender'])

# Step 6: Define features and target
X_combined = combined_data.drop(['customer_id', 'churn'], axis=1)
y_combined = combined_data['churn']

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Step 8: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
scaler_path = "C:/Users/Raghav/Desktop/Model/scaler_logreg.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved as {scaler_path}")

# Step 9: Train Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)

# Step 10: Evaluate model
y_pred = logreg_model.predict(X_test_scaled)
logreg_accuracy = accuracy_score(y_test, y_pred)
logreg_report = classification_report(y_test, y_pred)

# Save the model
model_path = "C:/Users/Raghav/Desktop/Model/logistic_regression_model.pkl"
joblib.dump(logreg_model, model_path)
print(f"Model saved as {model_path}")

print(f"Logistic Regression Model Accuracy: {logreg_accuracy * 100:.2f}%")
