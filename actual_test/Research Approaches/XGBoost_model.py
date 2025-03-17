# XGBoost Based churn model approach
# 70 percent accuuracy, might be dud


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load datasets
dataset1 = pd.read_csv("C:/Users/Raghav/Documents/GitHub/Predictive-Analytics-for-Customer-Retention/actual_test/Churn_Modelling.csv")
dataset2 = pd.read_csv("C:/Users/Raghav/Documents/GitHub/Predictive-Analytics-for-Customer-Retention/actual_test/Bank Customer Churn Prediction.csv")
dataset3 = pd.read_csv("C:/Users/Raghav/Documents/GitHub/Predictive-Analytics-for-Customer-Retention/actual_test/Customer-Churn-Records.csv")

# Step 2: Standardize column names
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

# Step 3: Select relevant columns
columns_to_use = [
    'customer_id', 'credit_score', 'country', 'gender', 'age', 'tenure', 'balance',
    'products_number', 'credit_card', 'active_member', 'estimated_salary', 'churn'
]
dataset1_filtered = dataset1[columns_to_use]
dataset2_filtered = dataset2[columns_to_use]
dataset3_filtered = dataset3[columns_to_use]

# Step 4: Combine datasets
combined_data = pd.concat([dataset1_filtered, dataset2_filtered, dataset3_filtered], ignore_index=True)

# Step 5: Encode categorical columns
le_country = LabelEncoder()
le_gender = LabelEncoder()
combined_data['country'] = le_country.fit_transform(combined_data['country'])
combined_data['gender'] = le_gender.fit_transform(combined_data['gender'])

# Step 6: Define features and target
X = combined_data.drop(columns=['customer_id', 'churn'])
y = combined_data['churn']

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
scaler_path = "C:/Users/Raghav/Desktop/Model/scaler_xgb.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved as {scaler_path}")

# Step 9: Train XGBoost model
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# Step 10: Evaluate model
y_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred)
xgb_report = classification_report(y_test, y_pred)

# Save the model
model_path = "C:/Users/Raghav/Desktop/Model/xgboost_model.pkl"
joblib.dump(xgb_model, model_path)
print(f"Model saved as {model_path}")

print(f"XGBoost Model Accuracy: {xgb_accuracy * 100:.2f}%")
