import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load the pre-trained model
model_filename = "optimized_random_forest_model.pkl"  # Make sure this file is in the same directory
model = joblib.load(model_filename)

# Step 2: Function to preprocess new input data
def preprocess_input(data):
    # Encode categorical variables
    le_country = LabelEncoder()
    le_gender = LabelEncoder()
    
    # Fit encoders based on known categories
    data['country'] = le_country.fit_transform(data['country'])
    data['gender'] = le_gender.fit_transform(data['gender'])

    # Feature scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data

# Step 3: Load new customer data to predict churn
# Create a sample DataFrame similar to the original training data
sample_data = pd.DataFrame({
    'credit_score': [600, 850],
    'country': ['France', 'Spain'],
    'gender': ['Male', 'Female'],
    'age': [35, 45],
    'tenure': [5, 3],
    'balance': [15000.00, 120000.00],
    'products_number': [2, 1],
    'credit_card': [1, 0],
    'active_member': [1, 1],
    'estimated_salary': [50000.00, 100000.00]
})

# Step 4: Preprocess the sample data
processed_data = preprocess_input(sample_data)

# Step 5: Make predictions using the loaded model
predictions = model.predict(processed_data)

# Step 6: Display predictions
for index, prediction in enumerate(predictions):
    result = "Churn" if prediction == 1 else "No Churn"
    print(f"Customer {index + 1}: {result}")
