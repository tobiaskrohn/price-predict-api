from flask import Flask, request, jsonify
import joblib
import pandas as pd  # <-- Import pandas

app = Flask(__name__)
model = joblib.load("price_model.pkl")  # Load your trained model

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Make sure the order and column names match your training data
    features = {
        "locality": data['locality'],
        "property_type": data['property_type'],
        "bedrooms": data['bedrooms'],
        "bathrooms": data['bathrooms'],
        "area": data['area'],
        "year_listed": data['year_listed'],
        "month_listed": data['month_listed'],
        "description": data['description']
    }

    # Wrap it in a DataFrame to preserve column names
    input_df = pd.DataFrame([features])

    # Predict
    prediction = model.predict(input_df)[0]

    return jsonify({"predicted_price": prediction})
