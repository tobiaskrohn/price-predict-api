import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import numpy as np

# Import our custom modules
from database import save_lead
from email_service import send_valuation_email

app = FastAPI(title="PropIQly Valuation API")

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Load the Model on Startup ---
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model_path = "model/price_model.pkl" 
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# --- 2. Define the Input Data Structure ---
class PropertyData(BaseModel):
    email: str
    locality: str
    property_type: str
    area: float
    bedrooms: int
    bathrooms: int
    description: Optional[str] = "No description provided."

# --- 3. The Valuation Endpoint ---
@app.post("/api/predict")
async def predict_price(data: PropertyData, background_tasks: BackgroundTasks):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    # A. Prepare Data for Model
    input_data = {
        "locality": [data.locality],
        "property_type": [data.property_type],
        "bedrooms": [data.bedrooms],
        "bathrooms": [data.bathrooms],
        "area": [data.area],
        "year_listed": [datetime.now().year],
        "month_listed": [datetime.now().month],
        "description": [data.description],
        "folder": ["web_request"]
    }
    
    # Add dummy image features
    for i in range(512):
        input_data[f"img_feat_{i}"] = [0.0]

    df = pd.DataFrame(input_data)

    # B. Run Prediction & ROUNDING
    try:
        raw_prediction = model.predict(df)[0]
        
        # ADJUSTMENT MADE HERE: 
        # round(x, -3) rounds to the nearest 1,000 (e.g., 345,600 -> 346,000)
        predicted_price = float(round(raw_prediction, -3))
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=400, detail="Internal prediction error.")

    # C. Background Tasks (using the rounded price)
    background_tasks.add_task(
        save_lead, 
        email=data.email, 
        locality=data.locality, 
        price=predicted_price, 
        area=data.area
    )
    
    background_tasks.add_task(
        send_valuation_email, 
        to_email=data.email, 
        price=predicted_price, 
        locality=data.locality
    )

    return {
        "status": "success",
        "predicted_price": predicted_price,
        "recipient": data.email
    }

@app.get("/")
def home():
    return {"status": "online", "engine": "PropIQly AI v1.0"}