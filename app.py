import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import re
import datetime
import resend
from supabase import create_client, Client

app = Flask(__name__)
CORS(app)

# Environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")

resend.api_key = RESEND_API_KEY
supabase = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Connected to Supabase")
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")

# Load model
model = None
try:
    model = joblib.load("price_model.pkl")
    print("✅ Model loaded successfully")
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
    return jsonify({
        "status": "online",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    try:
        data = request.get_json()
        
        email_to = data.get("email", "unknown").strip()
        locality = str(data.get("locality", "Sliema"))
        property_type = str(data.get("property_type", "Apartment"))
        area = clean_numeric(data.get("area"))
        bedrooms = clean_numeric(data.get("bedrooms"))
        bathrooms = clean_numeric(data.get("bathrooms"))
        description = str(data.get("description", "")).strip()
        intent = str(data.get("intent", "browser"))  # 'owner' or 'browser'

        if np.isnan(area) or area <= 0:
            return jsonify({"error": "Invalid area"}), 400

        # Prepare features for prediction
        input_dict = {
            "locality": locality,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "area": area,
            "year_listed": datetime.datetime.now().year,
            "month_listed": datetime.datetime.now().month,
            "description": description,
            "folder": "web_lead",
            "price_per_sqm": 0.0,
            "desc_length": len(description),
            "desc_word_count": len(description.split()),
            "bedrooms_per_area": bedrooms / area if area > 0 else 0,
            "bathrooms_per_area": bathrooms / area if area > 0 else 0,
            "bed_bath_ratio": bathrooms / bedrooms if bedrooms > 0 else 0,
        }
        # Fill image features (if your model expects them)
        for i in range(512):
            input_dict[f"img_feat_{i}"] = 0.0

        # Make prediction
        prediction_log_scale = model.predict(pd.DataFrame([input_dict]))[0]
        predicted_price = float(round(max(0, np.expm1(prediction_log_scale)), -3))

        # Send email (only place where price is revealed)
        if RESEND_API_KEY and email_to != "unknown":
            html_content = f"""
            <div style="font-family: sans-serif; max-width: 600px; margin: auto; padding: 40px; border: 1px solid #eee; border-radius: 24px;">
                <div style="text-align: center; margin-bottom: 30px;">
                    <h1 style="color: #4f46e5; font-size: 28px; font-weight: 900; letter-spacing: -1px;">PropIQly<span style="color: #4f46e5;">.</span></h1>
                </div>
                <div style="background-color: #f8fafc; padding: 30px; border-radius: 20px; text-align: center;">
                    <p style="text-transform: uppercase; font-size: 10px; font-weight: bold; color: #94a3b8; letter-spacing: 2px; margin-bottom: 10px;">Your Market Estimate</p>
                    <h2 style="font-size: 48px; color: #4f46e5; margin: 0; font-weight: 900;">€{predicted_price:,.0f}</h2>
                    <p style="color: #64748b; font-size: 14px; margin-top: 15px;">{property_type} in {locality}</p>
                </div>
                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #f1f5f9; text-align: center; color: #94a3b8; font-size: 12px;">
                    This is an AI-generated estimate. For a professional valuation or viewing, contact our team.
                </div>
            </div>
            """

            resend.Emails.send({
                "from": "PropIQly <no-reply@propiqly.com>",
                "to": email_to,
                "subject": "Your PropIQly Property Valuation",
                "html": html_content
            })

        # Save lead to Supabase
        if supabase:
            supabase.table("leads").insert({
                "email": email_to,
                "locality": locality,
                "property_type": property_type,
                "area_sqm": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "description": description,
                "intent": intent,
                "predicted_price": predicted_price,
                "created_at": datetime.datetime.utcnow().isoformat(),
                "source": "web_valuation"
            }).execute()

        # Never return the price to the frontend!
        return jsonify({"status": "success"})

    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))