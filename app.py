from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the scaler and the trained Keras model
scaler = joblib.load('C:/Users/Raghav/Desktop/Model/scaler.pkl')
model = load_model('C:/Users/Raghav/Desktop/Model/trained_neural_network_model.keras')

# Define categorical mappings for one-hot encoding
country_mapping = {'France': 0, 'Spain': 1, 'Germany': 2}
gender_mapping = {'Female': 0, 'Male': 1}

# Serve files from templates/ as if it were a static folder 
@app.route('/templates/<path:filename>')
def custom_static(filename):
    return send_from_directory('templates', filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        required_fields = ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary', 'country', 'gender']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        credit_score = float(data['credit_score'])
        age = float(data['age'])
        tenure = float(data['tenure'])
        balance = float(data['balance'])
        estimated_salary = float(data['estimated_salary'])

        country = data['country']
        gender = data['gender']

        country_one_hot = [0, 0, 0]
        if country in country_mapping:
            country_one_hot[country_mapping[country]] = 1

        gender_one_hot = [0, 0]
        if gender in gender_mapping:
            gender_one_hot[gender_mapping[gender]] = 1

        numeric_features = [credit_score, age, tenure, balance, estimated_salary]
        scaled_numeric = scaler.transform([numeric_features])[0]
        input_data = np.array([np.concatenate([scaled_numeric, country_one_hot, gender_one_hot])])

        prediction_prob = float(model.predict(input_data)[0][0])
        result = "Churn" if prediction_prob > 0.5 else "No Churn"

        return jsonify({
            "prediction": result,
            "probability": prediction_prob
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
