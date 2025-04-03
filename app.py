from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the scaler and the trained Keras model
scaler = joblib.load('C:/Users/Raghav/Desktop/Model/scaler.pkl')
model = load_model('C:/Users/Raghav/Desktop/Model/trained_neural_network_model.keras')

# Define categorical mappings for one-hot encoding
country_mapping = {'France': 0, 'Spain': 1, 'Germany': 2}
gender_mapping = {'Female': 0, 'Male': 1}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate required fields
        required_fields = ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary', 'country', 'gender']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Extract and convert numeric inputs
        credit_score = float(data['credit_score'])
        age = float(data['age'])
        tenure = float(data['tenure'])
        balance = float(data['balance'])
        estimated_salary = float(data['estimated_salary'])

        # Extract categorical inputs
        country = data['country']
        gender = data['gender']

        # One-hot encode country (France, Spain, Germany)
        country_one_hot = [0, 0, 0]
        if country in country_mapping:
            country_one_hot[country_mapping[country]] = 1

        # One-hot encode gender (Female, Male)
        gender_one_hot = [0, 0]
        if gender in gender_mapping:
            gender_one_hot[gender_mapping[gender]] = 1

        # Scale numeric inputs
        numeric_features = [credit_score, age, tenure, balance, estimated_salary]
        scaled_numeric = scaler.transform([numeric_features])[0]

        # Final input vector: (5 scaled + 3 country + 2 gender = 10)
        input_data = np.array([np.concatenate([scaled_numeric, country_one_hot, gender_one_hot])])

        # Make prediction
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
