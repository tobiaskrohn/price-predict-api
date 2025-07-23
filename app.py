import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import re
import json

app = Flask(__name__)

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
    cleaned = re.sub(r'[^\d.,]+', '', s_val).replace(',', '')
    return pd.to_numeric(cleaned, errors='coerce')

# Define the exact list of features used during training, INCLUDING DERIVED ONES.
# This list MUST EXACTLY match the 'available_features' used when training your model.
# The order is CRITICAL for the ColumnTransformer in your pipeline.
TRAINING_FEATURES_ORDER = [
    "locality",
    "property_type",
    "bedrooms",
    "bathrooms",
    "area",
    "desc_length",          # Derived
    "desc_word_count",      # Derived
    "bedrooms_per_area",    # Derived
    "bathrooms_per_area",   # Derived
    "bed_bath_ratio",       # Derived
    "description"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # --- 1. Basic Data Validation: Expected fields from Bubble.io ---
    # These are the ONLY fields you expect directly from the user/Bubble.io
    expected_user_input_fields = [
        "locality", "property_type", "bedrooms", "bathrooms", "area", "description"
    ]
    for field in expected_user_input_fields:
        if field not in data or data[field] is None:
            return jsonify({"error": f"Missing or null required field from user input: '{field}'"}), 400

    # --- 2. Clean Numeric Inputs ---
    # Use .get() with a default to avoid KeyError if Bubble sends unexpected None
    # though we've validated above.
    try:
        raw_area = data.get('area')
        raw_bedrooms = data.get('bedrooms')
        raw_bathrooms = data.get('bathrooms')

        processed_area = clean_numeric(raw_area)
        processed_bedrooms = clean_numeric(raw_bedrooms)
        processed_bathrooms = clean_numeric(raw_bathrooms)

    except Exception as e:
        return jsonify({"error": f"Error cleaning numeric input '{e}'. Check format for area, bedrooms, bathrooms."}), 400

    # --- Handle NaNs for crucial numerical inputs (post clean_numeric) ---
    if pd.isna(processed_area) or pd.isna(processed_bedrooms) or pd.isna(processed_bathrooms):
         return jsonify({"error": "Invalid numeric input detected (e.g., area, bedrooms, bathrooms could not be parsed). Please check values."}), 400

    # --- 3. Ensure positive values for 'area' ---
    if processed_area <= 0:
        return jsonify({"error": "Area must be a positive value."}), 400

    # --- 4. Apply Outlier Capping for 'area' using training statistics ---
    area_lower_bound = training_stats.get('area_lower_bound')
    area_upper_bound = training_stats.get('area_upper_bound')

    if area_lower_bound is None or area_upper_bound is None:
        return jsonify({"error": "Training statistics (area bounds) not loaded correctly."}), 500

    processed_area_capped = np.clip(processed_area, area_lower_bound, area_upper_bound)


    # --- 5. Feature Engineering (Calculate derived features from user input) ---
    # These calculations must match EXACTLY what was done during training
    description_text = data.get('description', '') # Safe default if description is empty/None for some reason

    desc_length = len(description_text)
    desc_word_count = len(description_text.split())

    # Use capped area for derived features to maintain consistency
    bedrooms_per_area = processed_bedrooms / (processed_area_capped + 1e-6)
    bathrooms_per_area = processed_bathrooms / (processed_area_capped + 1e-6)
    bed_bath_ratio = processed_bedrooms / (processed_bathrooms + 1e-6)


    # --- 6. Construct the FINAL input dictionary for the model ---
    # This dictionary must contain ALL features the model expects, both raw and derived.
    input_for_model = {
        "locality": data.get('locality'),
        "property_type": data.get('property_type'),
        "bedrooms": processed_bedrooms, # Use the cleaned/processed value
        "bathrooms": processed_bathrooms, # Use the cleaned/processed value
        "area": processed_area_capped, # Use the capped value
        "desc_length": desc_length,
        "desc_word_count": desc_word_count,
        "bedrooms_per_area": bedrooms_per_area,
        "bathrooms_per_area": bathrooms_per_area,
        "bed_bath_ratio": bed_bath_ratio,
        "description": description_text
    }

    # --- 7. Create DataFrame ensuring correct column order ---
    # This is CRITICAL. The ColumnTransformer relies on this order if it's not by name.
    # By specifying TRAINING_FEATURES_ORDER, we ensure consistency.
    try:
        input_df = pd.DataFrame([input_for_model])[TRAINING_FEATURES_ORDER]
    except KeyError as e:
        # This error means 'input_for_model' dictionary is missing a key expected by TRAINING_FEATURES_ORDER
        return jsonify({"error": f"Internal feature construction error: Missing data for '{e}'. This indicates a mismatch between API code and model training features."}), 500
    except Exception as e:
        return jsonify({"error": f"Error creating DataFrame for prediction: {e}"}), 500

    # --- 8. Predict and Inverse Transform ---
    # The loaded model (pipeline) automatically handles StandardScaler, OneHotEncoder, TfidfVectorizer
    try:
        prediction_log_scale = model.predict(input_df)[0]
        # Apply inverse log transformation to get the price in original scale
        predicted_price = np.expm1(prediction_log_scale)
        # Ensure price is non-negative, as expm1 can sometimes return tiny negative floats
        predicted_price = max(0, predicted_price)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}. Check input values and model integrity."}), 500

    # --- 9. Return Prediction ---
    return jsonify({"predicted_price": round(predicted_price, 2)}) # Round for cleaner output

if __name__ == "__main__":
    # Ensure 'price_model.pkl' and 'training_stats.json' are in the same directory
    # For production deployment, use a WSGI server like Gunicorn or uWSGI.
    # Example for local development:
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True is for dev only