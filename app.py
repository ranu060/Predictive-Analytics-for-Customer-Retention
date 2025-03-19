from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the scaler and the pickled model
scaler = joblib.load('scalar.pkl')
model = joblib.load('nueral-network.pkl')

# Define categorical mappings (adjust these if your encoding is different)
country_mapping = {'France': 0, 'Spain': 1, 'Germany': 2}
gender_mapping = {'Female': 0, 'Male': 1}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expected JSON payload:
        # {
        #   "credit_score": 600,
        #   "age": 40,
        #   "tenure": 3,
        #   "balance": 50000,
        #   "estimated_salary": 50000,
        #   "country": "France",
        #   "gender": "Female"
        # }
        data = request.get_json(force=True)
        credit_score = float(data.get('credit_score'))
        age = float(data.get('age'))
        tenure = float(data.get('tenure'))
        balance = float(data.get('balance'))
        estimated_salary = float(data.get('estimated_salary'))
        country = data.get('country')
        gender = data.get('gender')

        # Encode the categorical features
        country_encoded = country_mapping.get(country, 0)
        gender_encoded = gender_mapping.get(gender, 0)

        # Arrange input in the order: [credit_score, age, tenure, balance, estimated_salary, country, gender]
        input_data = np.array([[credit_score, age, tenure, balance, estimated_salary, country_encoded, gender_encoded]])

        # Scale the continuous features (first 5 columns)
        input_data[:, :5] = scaler.transform(input_data[:, :5])
        
        # Get prediction probability. If your model has predict_proba, use that;
        # otherwise, fall back to predict() and assume binary output.
        if hasattr(model, 'predict_proba'):
            prediction_prob = model.predict_proba(input_data)[0][1]
        else:
            prediction_prob = model.predict(input_data)[0]
        
        result = "Churn" if prediction_prob > 0.5 else "No Churn"

        return jsonify({
            "prediction": result,
            "probability": float(prediction_prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
