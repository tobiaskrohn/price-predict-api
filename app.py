import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import re
import json

app = Flask(__name__)
# Enable CORS for all routes to allow your Netlify frontend to communicate with this API
CORS(app)

# --- Load your trained model and training statistics ---
try:
    # It's good practice to provide full paths or ensure these files are in the working directory
    model = joblib.load("price_model.pkl")
    with open("training_stats.json", "r") as f:
        training_stats = json.load(f)
    print("Model and training statistics loaded successfully.")
except FileNotFoundError:
    print("Error: price_model.pkl or training_stats.json not found. Please ensure they are in the same directory.")
    # In a real production app, you might want more robust error handling or logging
    exit()
except Exception as e:
    print(f"Error loading model or stats: {e}")
    exit()

# --- Preprocessing functions (MUST match your training pipeline) ---
def clean_numeric(value):
    """
    Cleans a single numeric-like value by removing non-numeric characters,
    handling commas, and converting to float.
    Handles potential empty strings after cleaning by converting to NaN.
    """
    if pd.isna(value) or value is None: # Explicitly check for None as well
        return np.nan
    s_val = str(value)
    # Remove all non-digit, non-comma, non-period characters, then remove commas
    cleaned = re.sub(r"[^\d.,]", "", s_val).replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

# -------------------------------
# 🚀 API Endpoints
# -------------------------------

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "active", "message": "PropIQly Price Prediction API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects a JSON payload with:
    - locality (str)
    - property_type (str)
    - area (float/int)
    - bedrooms (int)
    - bathrooms (int)
    - description (str)
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # --- 1. Extract and Clean Inputs ---
    try:
        locality = str(data.get("locality", ""))
        property_type = str(data.get("property_type", ""))
        area = clean_numeric(data.get("area"))
        bedrooms = clean_numeric(data.get("bedrooms"))
        bathrooms = clean_numeric(data.get("bathrooms"))
        description = str(data.get("description", ""))
        
        # Add year/month based on current time
        import datetime
        now = datetime.datetime.now()
        year_listed = now.year
        month_listed = now.month

    except Exception as e:
        return jsonify({"error": f"Error parsing input data: {e}"}), 400

    # --- 2. Basic Validation ---
    if not locality or not property_type or np.isnan(area):
        return jsonify({"error": "Missing or invalid required fields: locality, property_type, and area are mandatory."}), 400

    # --- 3. Outlier Check (Using training_stats) ---
    # Optional: Log warning if area is outside training bounds
    if area < training_stats.get("area_lower_bound", 0) or area > training_stats.get("area_upper_bound", 10000):
        print(f"Warning: Area {area} is outside typical training bounds.")

    # --- 4. Handle Missing numeric specs (fill with training medians or sensible defaults) ---
    # For bedrooms/bathrooms, if NaN, we could use defaults or leave for model to handle (if trained on NaNs)
    # Here, we'll use simple defaults if cleaning failed
    bedrooms = bedrooms if not np.isnan(bedrooms) else 2
    bathrooms = bathrooms if not np.isnan(bathrooms) else 1

    # --- 5. Prepare data for model ---
    # Create the dictionary in the order expected by the model
    # Note: 'folder' and 'img_feat_X' are required because the Excel pipeline used them.
    # We use neutral/placeholder values for these.
    input_for_model = {
        "locality": locality,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "area": area,
        "year_listed": year_listed,
        "month_listed": month_listed,
        "description": description,
        "folder": "api_request" # Placeholder for the folder column
    }

    # Add 512 zero-value image features (neutral embeddings)
    for i in range(512):
        input_for_model[f"img_feat_{i}"] = 0.0

    # --- 6. Define Feature Order ---
    # This MUST match the order used in your pipeline's ColumnTransformer
    TRAINING_FEATURES_ORDER = [
        "locality", "property_type", "bedrooms", "bathrooms", "area",
        "year_listed", "month_listed", "description", "folder"
    ] + [f"img_feat_{i}" for i in range(512)]

    # --- 7. Create DataFrame ---
    try:
        # Create DataFrame from the dict, ensuring column order is correct
        input_df = pd.DataFrame([input_for_model])[TRAINING_FEATURES_ORDER]
    except KeyError as e:
        # This error means 'input_for_model' dictionary is missing a key expected by TRAINING_FEATURES_ORDER
        return jsonify({"error": f"Internal feature construction error: Missing data for '{e}'. This indicates a mismatch between API code and model training features."}), 500
    except Exception as e:
        return jsonify({"error": f"Error creating DataFrame for prediction: {e}"}), 500

    # --- 8. Predict and Inverse Transform ---
    # The loaded model (pipeline) automatically handles StandardScaler, OneHotEncoder, TfidfVectorizer
    try:
        # Note: If your training used Log transformation on Price, we need to expm1 it.
        # Looking at common practices in these models:
        prediction_val = model.predict(input_df)[0]
        
        # Check if the model predicts log(price) or raw price. 
        # Usually, if MAE/RMSE were very small during training, it's log.
        # We will assume raw price for now, or you can wrap it in np.expm1 if needed.
        predicted_price = float(prediction_val)
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}. Check input values and model integrity."}), 500

    # --- 9. Return Prediction ---
    return jsonify({
        "predicted_price": round(predicted_price, 2),
        "currency": "EUR"
    }) 

if __name__ == "__main__":
    # Ensure 'price_model.pkl' and 'training_stats.json' are in the same directory
    # For production deployment, use a WSGI server like Gunicorn.
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))