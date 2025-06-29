from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("price_model.pkl")  # load your trained model

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [
	data['locality'],
	data['property_type'],
        data['bedrooms'],
        data['bathrooms'],
        data['area'],
	data['year_listed']
	data['month_listed']
	data['description']
    ]
    prediction = model.predict([features])[0]
    return jsonify({"predicted_price": prediction})
