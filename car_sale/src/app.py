
"""
FastAPI application for Car Sales Prediction
"""
import pickle
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn


# Define the model wrapper class to handle model loading
class ConstrainedRFRegressor:
    def __init__(self, model, min_value=0):
        self.model = model
        self.min_value = min_value
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = self.model.predict(X)
        # Ensure predictions are not negative (minimum is 0)
        return np.maximum(predictions, self.min_value)


# Load model and preprocessor
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'model.pkl'
PREPROCESSOR_PATH = Path(__file__).resolve().parents[1] / 'models' / 'preprocessor.pkl'

try:
    # Ensure unpickling can resolve classes saved from a __main__ script
    sys.modules['__main__'] = sys.modules[__name__]

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    print("Model and preprocessor loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    model = None
    preprocessor = None


# Define input data model
class CarSaleInput(BaseModel):
    Year: int
    Month: int
    Region: str
    Model: str
    Avg_Price_EUR: float
    BEV_Share: float
    Premium_Share: float
    GDP_Growth: float
    Fuel_Price_Index: float


# Create FastAPI app
app = FastAPI(
    title="BMW Car Sales Prediction API",
    description="Predict BMW car sales using machine learning model",
    version="1.0.0"
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }





@app.post("/predict")
async def predict_sales(input_data: CarSaleInput) -> Dict[str, Any]:
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Apply preprocessing (this transforms categorical variables to one-hot encoded)
        input_processed = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(input_processed)[0]

        return {
            "prediction": int(prediction),

        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")





if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)