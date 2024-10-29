import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load your dataset (replace with your actual path if needed)
data = pd.read_csv('C:/Users/Raghav/Downloads/test/test/actual_test/Bank Customer Churn Prediction.csv')


# Drop 'customer_id' column as it's irrelevant for model training
data_cleaned = data.drop(columns=['customer_id'])

# Encode categorical variables 'country' and 'gender'
label_encoder_country = LabelEncoder()
label_encoder_gender = LabelEncoder()
data_cleaned['country'] = label_encoder_country.fit_transform(data_cleaned['country'])
data_cleaned['gender'] = label_encoder_gender.fit_transform(data_cleaned['gender'])

# Separate features (X) and target (y)
X = data_cleaned.drop(columns=['churn'])
y = data_cleaned['churn']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the continuous features (for 'credit_score', 'age', 'tenure', 'balance', 'estimated_salary')
scaler = StandardScaler()
X_train[['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']] = scaler.fit_transform(
    X_train[['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']]
)
X_test[['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']] = scaler.transform(
    X_test[['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']]
)

# Define the neural network model
model = tf.keras.models.Sequential()

# Input layer and first hidden layer
model.add(tf.keras.layers.Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Second hidden layer
model.add(tf.keras.layers.Dense(32, activation='relu'))

# Output layer (binary classification, so use sigmoid activation)
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save the trained model to a .h5 file
model.save('trained_neural_network_model.h5')

# Save the scaler to a .pkl file
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")
