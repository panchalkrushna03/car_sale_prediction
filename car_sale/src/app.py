# -*- coding: utf-8 -*-
"""
FastAPI application for Car Sales Prediction
"""
import pickle
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


@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with input form and prediction display"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BMW Car Sales Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { background-color: #3498db; color: white; padding: 12px 20px; border: none; border-radius: 5px; cursor: pointer; width: 100%; font-size: 16px; }
            button:hover { background-color: #2980b9; }
            .result { margin-top: 30px; padding: 20px; background-color: #ecf0f1; border-radius: 5px; }
            .prediction { font-size: 24px; font-weight: bold; color: #27ae60; text-align: center; }
            .examples { margin-top: 30px; }
            .example-btn { background-color: #95a5a6; margin: 5px; padding: 8px 15px; border: none; border-radius: 3px; cursor: pointer; }
            .example-btn:hover { background-color: #7f8c8d; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>BMW Car Sales Prediction</h1>

            <form id="predictionForm">
                <div class="form-group">
                    <label for="year">Year:</label>
                    <input type="number" id="year" name="Year" required min="2018" max="2030">
                </div>

                <div class="form-group">
                    <label for="month">Month:</label>
                    <input type="number" id="month" name="Month" required min="1" max="12">
                </div>

                <div class="form-group">
                    <label for="region">Region:</label>
                    <select id="region" name="Region" required>
                        <option value="">Select Region</option>
                        <option value="Europe">Europe</option>
                        <option value="China">China</option>
                        <option value="USA">USA</option>
                        <option value="Asia Pacific">Asia Pacific</option>
                        <option value="Other Markets">Other Markets</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="model">Model:</label>
                    <select id="model" name="Model" required>
                        <option value="">Select Model</option>
                        <option value="3 Series">3 Series</option>
                        <option value="5 Series">5 Series</option>
                        <option value="X3">X3</option>
                        <option value="X5">X5</option>
                        <option value="X7">X7</option>
                        <option value="i4">i4</option>
                        <option value="iX">iX</option>
                        <option value="MINI">MINI</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="avg_price">Average Price (EUR):</label>
                    <input type="number" id="avg_price" name="Avg_Price_EUR" required step="0.01">
                </div>

                <div class="form-group">
                    <label for="bev_share">BEV Share (%):</label>
                    <input type="number" id="bev_share" name="BEV_Share" required step="0.01" min="0" max="100">
                </div>

                <div class="form-group">
                    <label for="premium_share">Premium Share (%):</label>
                    <input type="number" id="premium_share" name="Premium_Share" required step="0.01" min="0" max="100">
                </div>

                <div class="form-group">
                    <label for="gdp_growth">GDP Growth (%):</label>
                    <input type="number" id="gdp_growth" name="GDP_Growth" required step="0.01">
                </div>

                <div class="form-group">
                    <label for="fuel_price">Fuel Price Index:</label>
                    <input type="number" id="fuel_price" name="Fuel_Price_Index" required step="0.01">
                </div>

                <button type="submit">Predict Sales</button>
            </form>

            <div id="result" class="result" style="display: none;">
                <h3>Prediction Result:</h3>
                <div class="prediction" id="prediction"></div>
            </div>

            <div class="examples">
                <h3>Example Data:</h3>
                <button class="example-btn" onclick="loadExample(0)">Example 1</button>
                <button class="example-btn" onclick="loadExample(1)">Example 2</button>
                <button class="example-btn" onclick="loadExample(2)">Example 3</button>
            </div>
        </div>

        <script>
            const examples = [
                {
                    Year: 2024,
                    Month: 6,
                    Region: "Europe",
                    Model: "3 Series",
                    Avg_Price_EUR: 45000,
                    BEV_Share: 15.5,
                    Premium_Share: 22.0,
                    GDP_Growth: 1.8,
                    Fuel_Price_Index: 1.15
                },
                {
                    Year: 2024,
                    Month: 6,
                    Region: "China",
                    Model: "X5",
                    Avg_Price_EUR: 55000,
                    BEV_Share: 25.0,
                    Premium_Share: 18.5,
                    GDP_Growth: 4.2,
                    Fuel_Price_Index: 1.08
                },
                {
                    Year: 2024,
                    Month: 6,
                    Region: "USA",
                    Model: "5 Series",
                    Avg_Price_EUR: 52000,
                    BEV_Share: 8.2,
                    Premium_Share: 25.0,
                    GDP_Growth: 2.1,
                    Fuel_Price_Index: 1.22
                }
            ];

            function loadExample(index) {
                const example = examples[index];
                document.getElementById('year').value = example.Year;
                document.getElementById('month').value = example.Month;
                document.getElementById('region').value = example.Region;
                document.getElementById('model').value = example.Model;
                document.getElementById('avg_price').value = example.Avg_Price_EUR;
                document.getElementById('bev_share').value = example.BEV_Share;
                document.getElementById('premium_share').value = example.Premium_Share;
                document.getElementById('gdp_growth').value = example.GDP_Growth;
                document.getElementById('fuel_price').value = example.Fuel_Price_Index;
            }

            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();

                const formData = new FormData(e.target);
                const data = Object.fromEntries(formData.entries());

                // Convert numeric fields
                data.Year = parseInt(data.Year);
                data.Month = parseInt(data.Month);
                data.Avg_Price_EUR = parseFloat(data.Avg_Price_EUR);
                data.BEV_Share = parseFloat(data.BEV_Share);
                data.Premium_Share = parseFloat(data.Premium_Share);
                data.GDP_Growth = parseFloat(data.GDP_Growth);
                data.Fuel_Price_Index = parseFloat(data.Fuel_Price_Index);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });

                    const result = await response.json();

                    if (response.ok) {
                        document.getElementById('prediction').textContent = `Predicted Units Sold: ${result.prediction.toLocaleString()}`;
                        document.getElementById('result').style.display = 'block';
                    } else {
                        alert('Error: ' + result.detail);
                    }
                } catch (error) {
                    alert('Error making prediction: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict")
async def predict_sales(input_data: CarSaleInput) -> Dict[str, Any]:
    """Predict car sales based on input features"""
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
            "input_data": input_data.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)