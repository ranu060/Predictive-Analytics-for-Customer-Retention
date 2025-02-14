import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("C:/Users/Raghav/Documents/GitHub/Predictive-Analytics-for-Customer-Retention/actual_test/Customer-Churn-Records-Final-Synthesized (1).csv")

# Handle Missing Values
df = df.dropna(thresh=len(df) * 0.7, axis=1)
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Remove Outliers using IQR
def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number])
    Q1 = numeric_cols.quantile(0.25)
    Q3 = numeric_cols.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]

df = remove_outliers(df)

# Feature Selection (Remove Low-Impact Features)
X = df.drop(columns=["target"])
y = df["target"]
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
selected_features = feature_importance[feature_importance > 0.01].index
df = df[selected_features]
df["target"] = y

# Remove Highly Correlated Features (Correlation > 0.9)
corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
df = df.drop(columns=to_drop)

# Variance Thresholding (Remove Low-Variance Features)
def remove_low_variance(data, threshold=0.01):
    selector = VarianceThreshold(threshold)
    filtered_data = selector.fit_transform(data)
    return pd.DataFrame(filtered_data, columns=data.columns[selector.get_support()])

df_filtered = remove_low_variance(df)

# Balance Data with SMOTE (If Needed)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(df_filtered.drop(columns=["target"]), df_filtered["target"])
df_balanced = pd.DataFrame(X_resampled, columns=df_filtered.columns[:-1])
df_balanced["target"] = y_resampled

# Add Gaussian Noise for Data Augmentation
noise_factor = 0.01
for col in df_balanced.select_dtypes(include=[np.number]).columns:
    df_balanced[col] += np.random.normal(loc=0, scale=noise_factor * df_balanced[col].std(), size=df_balanced.shape[0])

# Encode Categorical Features
df_final = pd.get_dummies(df_balanced, drop_first=True)

# Save Processed Dataset
df_final.to_csv("processed_dataset.csv", index=False)

print("Data processing complete! New dataset shape:", df_final.shape)