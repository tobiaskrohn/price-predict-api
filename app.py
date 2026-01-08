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
# Enable CORS so your Netlify site can talk to this API
CORS(app)

# --- 1. Supabase Connection ---
# These pull from the Environment Variables you set in Render
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
    with open("training_stats.json", "r") as f:
        training_stats = json.load(f)
    print("✅ Model and stats loaded.")
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

# --- 4. API Endpoints ---

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "online", "database": "connected" if supabase else "disconnected"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Extract user inputs
    email = data.get("email", "unknown@example.com")
    locality = str(data.get("locality", "Sliema"))
    property_type = str(data.get("property_type", "Apartment"))
    area = clean_numeric(data.get("area"))
    bedrooms = clean_numeric(data.get("bedrooms"))
    bathrooms = clean_numeric(data.get("bathrooms"))
    description = str(data.get("description", ""))

    if np.isnan(area):
        return jsonify({"error": "Invalid area provided"}), 400

    # --- Prepare Data for Model ---
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

    # Add 512 image features (zeros) as expected by your specific model pipeline
    for i in range(512):
        input_dict[f"img_feat_{i}"] = 0.0

    # Create DataFrame in correct order
    TRAINING_FEATURES_ORDER = [
        "locality", "property_type", "bedrooms", "bathrooms", "area",
        "year_listed", "month_listed", "description", "folder"
    ] + [f"img_feat_{i}" for i in range(512)]
    
    input_df = pd.DataFrame([input_dict])[TRAINING_FEATURES_ORDER]

    # --- Run Prediction ---
    try:
        prediction_log_scale = model.predict(input_df)[0]
        # Your model uses log scale: np.expm1 converts it back to Euro
        predicted_price = np.expm1(prediction_log_scale)
        predicted_price = max(0, float(predicted_price))
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # --- 5. Save Lead to Supabase ---
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
            print(f"✅ Lead saved for {email}")
        except Exception as e:
            print(f"❌ Database save error: {e}")

    return jsonify({
        "predicted_price": round(predicted_price, 2),
        "currency": "EUR"
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))