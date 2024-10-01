
import pandas as pd
from pymongo import MongoClient

# Function to connect to MongoDB
def connect_to_mongodb(uri="mongodb://localhost:27017/", db_name="churn_db", collection_name="customers"):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    return collection

# Function to load CSV data into MongoDB
def load_csv_to_mongodb(csv_file_path, collection):
    data = pd.read_csv(csv_file_path)
    # Insert data into MongoDB
    collection.insert_many(data.to_dict("records"))

# Function to clean and preprocess data for modeling
def preprocess_data(collection):
    # Fetch data from MongoDB
    data = pd.DataFrame(list(collection.find()))
    
    # Example preprocessing (this would depend on your dataset)
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    
    # Convert categorical variables to numeric (One-Hot Encoding)
    data = pd.get_dummies(data)
    
    return data

if __name__ == "__main__":
    # Example usage
    collection = connect_to_mongodb()
    load_csv_to_mongodb("path_to_csv_file.csv", collection)
    processed_data = preprocess_data(collection)
    print(processed_data.head())  # Display preprocessed data
