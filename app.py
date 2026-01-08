import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import re
import json
import datetime
from supabase import create_client, Client

app = Flask(__name__)
# Enable CORS for all routes (allows Netlify frontend to connect)
CORS(app)

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
    # Load model. Scikit-learn handles the pipeline logic.
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
        "database": "connected" if supabase else "disconnected",
        "features": "engineered"
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

    # Defaults
    bedrooms = bedrooms if not np.isnan(bedrooms) else 2
    bathrooms = bathrooms if not np.isnan(bathrooms) else 1

    # --- Prepare Data ---
    now = datetime.datetime.now()
    
    # Base dictionary
    input_dict = {
        "locality": locality,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "area": area,
        "year_listed": now.year,
        "month_listed": now.month,
        "description": description,
        "folder": "web_lead"
    }

    # --- Feature Engineering (Matches Training) ---
    # These were missing before, causing the model crash
    input_dict['desc_length'] = len(description) if description else 0
    input_dict['desc_word_count'] = len(description.split()) if description else 0
    
    safe_area = area if area > 0 else 1
    input_dict['bedrooms_per_area'] = bedrooms / safe_area
    input_dict['bathrooms_per_area'] = bathrooms / safe_area
    
    safe_beds = bedrooms if bedrooms > 0 else 1
    input_dict['bed_bath_ratio'] = bathrooms / safe_beds

    # Add 512 dummy image features (zeros)
    for i in range(512):
        input_dict[f"img_feat_{i}"] = 0.0

    # Create DataFrame
    # Note: Scikit-learn pipelines are usually robust to column order if passed a DataFrame with names,
    # but we pass the dict to DataFrame directly to let pandas handle the schema.
    input_df = pd.DataFrame([input_dict])

    # --- Run Prediction ---
    try:
        prediction_log_scale = model.predict(input_df)[0]
        # Inverse log transformation (np.expm1)
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
                "bedrooms": int(bedrooms),
                "bathrooms": int(bathrooms)
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