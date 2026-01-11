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
    return jsonify({"status": "online"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    
    # --- DATA EXTRACTION (Fixing Nulls) ---
    # We use data.get() for every field to ensure we capture what the frontend sends.
    # Note: Frontend sends 'email' (not email_to), so we map it here.
    email_to = data.get('email') 
    locality = data.get('locality')
    prop_type = data.get('property_type')
    
    # Clean numeric fields (handle strings like "2" or "100")
    area = clean_numeric(data.get('area'))
    bedrooms = clean_numeric(data.get('bedrooms'))
    bathrooms = clean_numeric(data.get('bathrooms'))
    
    # New fields
    is_owner = data.get('is_owner', False) # Default to False if missing
    description = data.get('description', '')

    # --- Model Input Preparation ---
    input_dict = {
        'locality': locality,
        'property_type': prop_type,
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
    }
    
    safe_beds = bedrooms if bedrooms > 0 else 1
    input_dict['bed_bath_ratio'] = bathrooms / safe_beds
    for i in range(512): input_dict[f"img_feat_{i}"] = 0.0

    try:
        input_df = pd.DataFrame([input_dict])
        pred_log = model.predict(input_df)[0]
        predicted_price = float(round(max(0, np.expm1(pred_log)), -3))

        # --- 1. Send Email ---
        if RESEND_API_KEY and email_to:
            html_content = f"""
            <div style="font-family: sans-serif; padding: 20px; color: #1e293b;">
                <h1 style="color: #4f46e5;">PropIQly Valuation</h1>
                <p>Your property estimate is ready:</p>
                <div style="background: #f8fafc; padding: 30px; border-radius: 20px; text-align: center; border: 1px solid #e2e8f0;">
                    <h2 style="font-size: 48px; color: #4f46e5; margin: 0;">€{predicted_price:,.0f}</h2>
                    <p style="color: #64748b;">{prop_type} in {locality}</p>
                </div>
            </div>
            """
            resend.Emails.send({
                "from": "PropIQly <no-reply@propiqly.com>",
                "to": email_to,
                "subject": "Your Property Valuation Result",
                "html": html_content
            })

        # --- 2. Save to Supabase (Explicit Mapping) ---
        if supabase:
            # We explicitly cast to ensure Postgres accepts the values
            db_record = {
                "email": email_to, 
                "locality": locality, 
                "property_type": prop_type,
                "area": area, 
                "bedrooms": int(bedrooms) if not np.isnan(bedrooms) else 0,
                "bathrooms": int(bathrooms) if not np.isnan(bathrooms) else 0,
                "predicted_price": predicted_price,
                "is_owner": bool(is_owner),      
                "description": str(description)
            }
            supabase.table("leads").insert(db_record).execute()

        return jsonify({"predicted_price": predicted_price}), 200

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)