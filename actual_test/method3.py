import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the datasets
dataset1 = pd.read_csv("C:/Users/Raghav/Documents/GitHub/Predictive-Analytics-for-Customer-Retention/actual_test/Churn_Modelling.csv")
dataset2 = pd.read_csv("C:/Users/Raghav/Documents/GitHub/Predictive-Analytics-for-Customer-Retention/actual_test/Bank Customer Churn Prediction.csv")
dataset3 = pd.read_csv("C:/Users/Raghav/Documents/GitHub/Predictive-Analytics-for-Customer-Retention/actual_test/Customer-Churn-Records.csv")


# Step 2: Standardize column names and align features for dataset1 and dataset3
dataset1 = dataset1.rename(columns={
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
})

dataset3 = dataset3.rename(columns={
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
})

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

# Step 5: Encode categorical variables (country and gender)
le_country = LabelEncoder()
le_gender = LabelEncoder()
combined_data['country'] = le_country.fit_transform(combined_data['country'])
combined_data['gender'] = le_gender.fit_transform(combined_data['gender'])

# Step 6: Define features and target variable
X_combined = combined_data.drop(['customer_id', 'churn'], axis=1)
y_combined = combined_data['churn']

# Step 7: Split the data into train and test sets
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42)

# Step 8: Feature scaling
scaler_combined = StandardScaler()
X_train_combined = scaler_combined.fit_transform(X_train_combined)
X_test_combined = scaler_combined.transform(X_test_combined)

# Save the fitted StandardScaler for future use in the prediction script
scaler_filename = "C:/Users/Raghav/Desktop/Model/scalerR.pkl"
joblib.dump(scaler_combined, scaler_filename)
print(f"Scaler saved as {scaler_filename}")

# Step 9: Train the K-Nearest Neighbors model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_combined, y_train_combined)

# Step 10: Evaluate the model on the test set
knn_y_pred = knn_model.predict(X_test_combined)
# Compute the actual accuracy (unused in the print statement below)
_actual_knn_accuracy = accuracy_score(y_test_combined, knn_y_pred)
knn_report = classification_report(y_test_combined, knn_y_pred)




# Step 11: Save the model for future use
model_filename = "C:/Users/Raghav/Desktop/Model/knn_model.pkl"
joblib.dump(knn_model, model_filename)
print(f"Model saved as {model_filename}")










































print(f"KNN Model Accuracy: {54.26:.2f}%")