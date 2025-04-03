from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model  # ✅ Add this

app = Flask(__name__)

# Load the scaler and the Keras model correctly
scaler = joblib.load('C:/Users/Raghav/Desktop/Model/scaler.pkl')
model = load_model('C:/Users/Raghav/Desktop/Model/trained_neural_network_model.keras')

# Define categorical mappings
country_mapping = {'France': 0, 'Spain': 1, 'Germany': 2}
gender_mapping = {'Female': 0, 'Male': 1}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Input validation
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

        # Encode categorical variables
        country_encoded = country_mapping.get(country, 0)
        gender_encoded = gender_mapping.get(gender, 0)

        # Prepare input data
        input_data = np.array([[credit_score, age, tenure, balance, estimated_salary, country_encoded, gender_encoded]])

        # Scale only the continuous features (first 5)
        input_data[:, :5] = scaler.transform(input_data[:, :5])

        # Predict using the Keras model
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
