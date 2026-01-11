import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import re
import json
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "online"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        email_to = data.get('email')
        locality = data.get('locality')
        property_type = data.get('property_type')
        area = float(data.get('area', 0))
        bedrooms = float(data.get('bedrooms', 0))
        bathrooms = float(data.get('bathrooms', 0))
        is_owner = data.get('is_owner', True)
        description = data.get('description', "")

        # --- 3. Feature Engineering (Crucial for your model) ---
        desc_text = str(description) if description else ""
        
        input_dict = {
            'locality': locality,
            'property_type': property_type,
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'description': desc_text,
            'desc_length': len(desc_text),
            'desc_word_count': len(desc_text.split()),
            'bedrooms_per_area': bedrooms / area if area > 0 else 0,
            'bathrooms_per_area': bathrooms / area if area > 0 else 0
        }

        # Add dummy image features if your model expects them (512 features)
        for i in range(512):
            input_dict[f"img_feat_{i}"] = 0.0

        # Create DataFrame
        input_df = pd.DataFrame([input_dict])

        # --- 4. Prediction ---
        prediction_log = model.predict(input_df)[0]
        predicted_price = float(round(np.expm1(prediction_log), -3))

        # --- 5. Send Email ---
        if RESEND_API_KEY:
            html_content = f"""
            <div style="font-family: sans-serif; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #eee; border-radius: 10px;">
                <h1 style="color: #4f46e5;">PropIQly Valuation</h1>
                <p>Your property estimate is ready:</p>
                <div style="background: #f8fafc; padding: 30px; border-radius: 20px; text-align: center; border: 1px solid #e2e8f0;">
                    <h2 style="font-size: 48px; color: #4f46e5; margin: 0;">€{predicted_price:,.0f}</h2>
                    <p style="color: #64748b;">{property_type} in {locality}</p>
                </div>
            </div>
            """
            resend.Emails.send({
                "from": "PropIQly <no-reply@propiqly.com>",
                "to": email_to,
                "subject": "Your Property Valuation Result",
                "html": html_content
            })

        # --- 6. Save Lead ---
        if supabase:
            supabase.table("leads").insert({
                "email": email_to, 
                "locality": locality, 
                "property_type": property_type,
                "area": area,
                "bedrooms": int(bedrooms),
                "bathrooms": int(bathrooms),
                "predicted_price": predicted_price,
                "is_owner": is_owner,
                "description": description
            }).execute()

        return jsonify({"predicted_price": predicted_price}), 200

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)