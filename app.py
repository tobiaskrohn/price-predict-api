import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # <--- CRITICAL IMPORT
import joblib
import pandas as pd
import numpy as np
import re
import json
import datetime
from supabase import create_client, Client

app = Flask(__name__)
CORS(app) # <--- CRITICAL: Enables Cross-Origin Requests

# ... (Rest of your backend logic stays exactly the same) ...
# Copy the rest of the logic from your previous app.py file here.
# For simplicity in this file block, I am only showing the CORS part.
# Ensure you keep the Supabase and Model loading logic.

# --- 1. Supabase Connection ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connection initialized.")
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")

# --- 2. Load Model and Stats ---
try:
    model = joblib.load("price_model.pkl")
    if os.path.exists("training_stats.json"):
        with open("training_stats.json", "r") as f:
            training_stats = json.load(f)
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- 3. Preprocessing Helpers ---
def clean_numeric(value):
    if pd.isna(value) or value is None:
        return np.nan
    s_val = str(value)
    cleaned = re.sub(r"[^\d.,]", "", s_val).replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "online", 
        "database": "connected" if supabase else "disconnected"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Extract inputs
    email = data.get("email", "unknown")
    locality = str(data.get("locality", "Sliema"))
    property_type = str(data.get("property_type", "Apartment"))
    area = clean_numeric(data.get("area"))
    bedrooms = clean_numeric(data.get("bedrooms"))
    bathrooms = clean_numeric(data.get("bathrooms"))
    description = str(data.get("description", ""))

    if np.isnan(area):
        return jsonify({"error": "Invalid area provided"}), 400

    # --- Prepare Data ---
    now = datetime.datetime.now()
    input_dict = {
        "locality": locality,
        "property_type": property_type,
        "bedrooms": bedrooms if not np.isnan(bedrooms) else 2,
        "bathrooms": bathrooms if not np.isnan(bathrooms) else 1,
        "area": area,
        "year_listed": now.year,
        "month_listed": now.month,
        "description": description,
        "folder": "web_lead"
    }

    # Add 512 dummy image features
    for i in range(512):
        input_dict[f"img_feat_{i}"] = 0.0

    # Create DataFrame
    TRAINING_FEATURES_ORDER = [
        "locality", "property_type", "bedrooms", "bathrooms", "area",
        "year_listed", "month_listed", "description", "folder"
    ] + [f"img_feat_{i}" for i in range(512)]
    
    input_df = pd.DataFrame([input_dict])[TRAINING_FEATURES_ORDER]

    # --- Run Prediction ---
    try:
        prediction_log_scale = model.predict(input_df)[0]
        # Inverse log transformation
        predicted_price = np.expm1(prediction_log_scale)
        predicted_price = max(0, float(predicted_price))
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # --- Save Lead to Supabase ---
    if supabase:
        try:
            lead_data = {
                "email": email,
                "locality": locality,
                "property_type": property_type,
                "area": area,
                "predicted_price": round(predicted_price, 2),
                "bedrooms": int(input_dict["bedrooms"]),
                "bathrooms": int(input_dict["bathrooms"])
            }
            supabase.table("leads").insert(lead_data).execute()
        except Exception as e:
            print(f"Database Error: {e}")

    return jsonify({
        "predicted_price": round(predicted_price, 2),
        "currency": "EUR"
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))