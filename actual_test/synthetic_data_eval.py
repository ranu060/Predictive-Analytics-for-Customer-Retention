import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Step 1: Load the trained model and the saved scaler
model_filename = "optimized_random_forest_model.pkl"
scaler_filename = "scaler.pkl"
model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Step 2: Create a large synthetic dataset with random values
# Generating 1000 synthetic samples with similar feature distributions
np.random.seed(42)  # For reproducibility

synthetic_data = {
    'credit_score': np.random.randint(300, 850, size=1000),         # Random credit scores between 300 and 850
    'country': np.random.randint(0, 3, size=1000),                  # Randomly assigning encoded country values
    'gender': np.random.randint(0, 2, size=1000),                   # Randomly assigning encoded gender values (0 or 1)
    'age': np.random.randint(18, 80, size=1000),                    # Age between 18 and 80
    'tenure': np.random.randint(0, 10, size=1000),                  # Tenure between 0 and 10 years
    'balance': np.random.uniform(0, 200000, size=1000),             # Random balance between 0 and 200,000
    'products_number': np.random.randint(1, 5, size=1000),          # Random number of products (1 to 4)
    'credit_card': np.random.randint(0, 2, size=1000),              # Random binary for credit card ownership
    'active_member': np.random.randint(0, 2, size=1000),            # Random binary for active membership
    'estimated_salary': np.random.uniform(20000, 150000, size=1000) # Random salary between 20,000 and 150,000
}

# Step 3: Convert to DataFrame and preprocess using the same scaler
synthetic_df = pd.DataFrame(synthetic_data)

# Standardize the features using the saved scaler
synthetic_df_scaled = scaler.transform(synthetic_df)

# Step 4: Run the predictions using the loaded model
synthetic_predictions = model.predict(synthetic_df_scaled)

# Step 5: Create synthetic true labels for evaluation (simulate 50% churn for testing)
# For the sake of this synthetic evaluation, we'll assume a 50-50 distribution of churn vs non-churn
synthetic_true_labels = np.random.randint(0, 2, size=1000)

# Step 6: Calculate metrics
accuracy = accuracy_score(synthetic_true_labels, synthetic_predictions)
precision = precision_score(synthetic_true_labels, synthetic_predictions)
recall = recall_score(synthetic_true_labels, synthetic_predictions)
f1 = f1_score(synthetic_true_labels, synthetic_predictions)
classification_rep = classification_report(synthetic_true_labels, synthetic_predictions)

# Display the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")
print("Classification Report:")
print(classification_rep)
