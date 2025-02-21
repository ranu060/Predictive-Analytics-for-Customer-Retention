import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from imblearn.combine import SMOTETomek
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load dataset
df = pd.read_csv("C:/Users/Raghav/Documents/GitHub/Predictive-Analytics-for-Customer-Retention/actual_test/Customer-Churn-Records-Final-Synthesized (1).csv")

# Define the correct target column
target_col = "Exited"  # Updated from "target" to "Exited"

# Ensure the target column exists
if target_col not in df.columns:
    raise KeyError(f"Column '{target_col}' not found in dataset. Available columns: {df.columns.tolist()}")

# Drop unnecessary ID columns that don't contribute to predictions
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")

# Separate numeric and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

# Apply KNN Imputer only to numeric columns
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])

# Encode Categorical Features
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Remove Outliers using Z-score
df = df[(np.abs(zscore(df[num_cols])) < 3).all(axis=1)]

# Ensure target column exists before dropping it
if target_col not in df.columns:
    raise KeyError(f"'{target_col}' column is missing after preprocessing. Check previous steps.")

# Feature Selection using Mutual Information
X = df.drop(columns=[target_col])
y = df[target_col]
mi_scores = mutual_info_classif(X, y)
selected_features = X.columns[mi_scores > 0.01].tolist()
df = df[selected_features + [target_col]]

# Remove Highly Correlated Features using VIF
def remove_multicollinearity(data, threshold=5.0):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return data[vif_data[vif_data["VIF"] < threshold]["Feature"]]

df = remove_multicollinearity(df)

# Balance Data with SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(df.drop(columns=[target_col]), df[target_col])
df_balanced = pd.DataFrame(X_resampled, columns=df.drop(columns=[target_col]).columns)
df_balanced[target_col] = y_resampled

# Save Processed Dataset
df_balanced.to_csv("C:/Users/Raghav/Desktop/Model/processed_dataset_optimized.csv", index=False)

print("Data processing complete! New dataset shape:", df_balanced.shape)
