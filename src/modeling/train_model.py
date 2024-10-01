
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to connect to MongoDB and load data
def load_data_from_mongodb(uri="mongodb://localhost:27017/", db_name="churn_db", collection_name="customers"):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    data = pd.DataFrame(list(collection.find()))
    return data

# Train the model using RandomForest
def train_churn_model():
    data = load_data_from_mongodb()
    
    # Assume 'Churn' is the target variable, and we drop '_id' and 'Churn' for features
    X = data.drop(columns=['_id', 'Churn'])
    y = data['Churn']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    
    return model

if __name__ == "__main__":
    trained_model = train_churn_model()
