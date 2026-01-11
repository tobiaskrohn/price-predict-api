import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import re
import json
import datetime
import resend 
from supabase import create_client, Client

app = Flask(__name__)
CORS(app)

# --- 1. Connections ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")

resend.api_key = RESEND_API_KEY
supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")

# --- 2. Load Model ---
model = None
try:
    model = joblib.load("price_model.pkl")
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def clean_numeric(value):
    if pd.isna(value) or value is None:
        return np.nan
    s_val = str(value)
    cleaned = re.sub(r"[^\d.,]", "", s_val)
    if not cleaned:
        return np.nan
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except:
        return np.nan

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "online", "model_loaded": model is not None}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded on server"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Extract new fields alongside existing ones
    email_to = data.get('email')
    locality = data.get('locality', 'Sliema')
    property_type = data.get('property_type', 'Apartment')
    area = clean_numeric(data.get('area', 100))
    bedrooms = clean_numeric(data.get('bedrooms', 2))
    bathrooms = clean_numeric(data.get('bathrooms', 2))
    
    # New Fields
    is_owner = data.get('is_owner', True) # Boolean
    description = data.get('description', '') # String

    if not email_to:
        return jsonify({"error": "Email is required"}), 400

    # Prepare features for ML Model
    input_dict = {
        'locality': locality,
        'property_type': property_type,
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
    }

    # Derived feature for model
    safe_beds = bedrooms if bedrooms > 0 else 1
    input_dict['bed_bath_ratio'] = bathrooms / safe_beds

    # Add dummy image features if model expects them
    for i in range(512):
        input_dict[f"img_feat_{i}"] = 0.0

    input_df = pd.DataFrame([input_dict])

    try:
        prediction_log_scale = model.predict(input_df)[0]
        raw_price = np.expm1(prediction_log_scale)
        predicted_price = float(round(max(0, raw_price), -3))
        
        # --- Email Alert via Resend ---
        if RESEND_API_KEY:
            html_content = f"""
            <div style="font-family: sans-serif; max-width: 600px; margin: auto; border: 1px solid #f1f5f9; border-radius: 24px; padding: 40px;">
                <h1 style="color: #0f172a; font-size: 24px; font-weight: 900; margin-bottom: 8px;">Valuation Ready</h1>
                <p style="color: #64748b; margin-bottom: 30px;">Based on our AI analysis, here is the market estimate for your property.</p>
                <div style="background: #f8fafc; border-radius: 20px; padding: 30px; text-align: center;">
                    <p style="text-transform: uppercase; font-size: 10px; font-weight: bold; color: #94a3b8; letter-spacing: 2px; margin-bottom: 10px;">Market Estimate</p>
                    <h2 style="font-size: 48px; color: #4f46e5; margin: 0; font-weight: 900;">€{predicted_price:,.0f}</h2>
                    <p style="color: #64748b; font-size: 14px; margin-top: 15px;">Locality: <b>{locality}</b> | Type: <b>{property_type}</b></p>
                    <p style="color: #94a3b8; font-size: 12px; margin-top: 5px;">Inquiry Type: <b>{"Property Owner" if is_owner else "Listing Sense Check"}</b></p>
                </div>
                <div style="margin-top: 30px; border-top: 1px solid #f1f5f9; padding-top: 20px; text-align: center; color: #94a3b8; font-size: 12px;">
                    This estimate is powered by PropIQly AI.
                </div>
            </div>
            """
            resend.Emails.send({
                "from": "PropIQly <no-reply@propiqly.com>",
                "to": email_to,
                "subject": "Your Property Valuation Result",
                "html": html_content
            })

        # --- Save to Supabase (Updated with new fields) ---
        if supabase:
            supabase.table("leads").insert({
                "email": email_to, 
                "locality": locality, 
                "property_type": property_type,
                "area": area,
                "bedrooms": int(bedrooms),
                "bathrooms": int(bathrooms),
                "predicted_price": predicted_price,
                "is_owner": is_owner,      # Ensure this column exists in DB
                "description": description # Ensure this column exists in DB
            }).execute()

        return jsonify({"predicted_price": predicted_price}), 200

    except Exception as e:
        print(f"Prediction/Storage Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)