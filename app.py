import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import re
import json
import datetime
import resend  # Added Resend library
from supabase import create_client, Client

app = Flask(__name__)
CORS(app)

# --- 1. Connections ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY") # Add this to Render Env Vars

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
    cleaned = re.sub(r"[^\d.,]", "", s_val).replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "online", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    data = request.get_json()
    email_to = data.get("email", "unknown")
    locality = str(data.get("locality", "Sliema"))
    property_type = str(data.get("property_type", "Apartment"))
    area = clean_numeric(data.get("area"))
    bedrooms = clean_numeric(data.get("bedrooms"))
    bathrooms = clean_numeric(data.get("bathrooms"))
    description = str(data.get("description", ""))

    if np.isnan(area):
        return jsonify({"error": "Invalid area"}), 400

    # Prediction Logic
    input_dict = {
        "locality": locality, "property_type": property_type,
        "bedrooms": bedrooms, "bathrooms": bathrooms, "area": area,
        "year_listed": datetime.datetime.now().year,
        "month_listed": datetime.datetime.now().month,
        "description": description, "folder": "web_lead", "price_per_sqm": 0.0
    }
    input_dict['desc_length'] = len(description)
    input_dict['desc_word_count'] = len(description.split())
    input_dict['bedrooms_per_area'] = bedrooms / (area if area > 0 else 1)
    input_dict['bathrooms_per_area'] = bathrooms / (area if area > 0 else 1)
    input_dict['bed_bath_ratio'] = bathrooms / (bedrooms if bedrooms > 0 else 1)
    for i in range(512): input_dict[f"img_feat_{i}"] = 0.0

    try:
        prediction_log_scale = model.predict(pd.DataFrame([input_dict]))[0]
        predicted_price = float(round(max(0, np.expm1(prediction_log_scale)), -3))
        
        # --- Send Email via Resend ---
        if RESEND_API_KEY:
            html_content = f"""
            <div style="font-family: sans-serif; max-width: 600px; margin: auto; padding: 40px; border: 1px solid #eee; border-radius: 24px;">
                <div style="text-align: center; margin-bottom: 30px;">
                    <h1 style="color: #4f46e5; font-size: 28px; font-weight: 900; letter-spacing: -1px;">PropIQly<span style="color: #4f46e5;">.</span></h1>
                </div>
                <div style="background-color: #f8fafc; padding: 30px; border-radius: 20px; text-align: center;">
                    <p style="text-transform: uppercase; font-size: 10px; font-weight: bold; color: #94a3b8; letter-spacing: 2px; margin-bottom: 10px;">Market Estimate</p>
                    <h2 style="font-size: 48px; color: #4f46e5; margin: 0; font-weight: 900;">€{predicted_price:,.0f}</h2>
                    <p style="color: #64748b; font-size: 14px; margin-top: 15px;">Locality: <b>{locality}</b> | Type: <b>{property_type}</b></p>
                </div>
                <div style="margin-top: 30px; border-top: 1px solid #f1f5f9; padding-top: 20px; text-align: center; color: #94a3b8; font-size: 12px;">
                    This estimate is powered by PropIQly AI. For a detailed human inspection, please contact our experts.
                </div>
            </div>
            """
            resend.Emails.send({
                "from": "PropIQly <no-reply@propiqly.com>",
                "to": email_to,
                "subject": "Your Property Valuation Result",
                "html": html_content
            })

        # Save to DB
        if supabase:
            supabase.table("leads").insert({"email": email_to, "locality": locality, "predicted_price": predicted_price, "property_type": property_type, "area": area, "bedrooms": int(bedrooms)}).execute()

        return jsonify({"status": "success"}) # Don't return price to frontend

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":

    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

