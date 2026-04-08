#!/usr/bin/env python3
"""
Test script for the Car Sales Prediction API
"""
import requests
import json

# Load example data
with open('example_data.json', 'r') as f:
    examples = json.load(f)

def test_prediction(example_idx=0):
    """Test the prediction endpoint with example data"""
    example = examples[example_idx]

    url = "http://127.0.0.1:8000/predict"

    try:
        response = requests.post(url, json=example)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"Input: {example}")
            print(f"Predicted units sold: {result['prediction']:,}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(response.text)
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the app is running with: python src/app.py")
        return False

def test_health():
    """Test the health check endpoint"""
    url = "http://127.0.0.1:8000/health"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check successful!")
            print(f"Status: {result['status']}")
            print(f"Model loaded: {result['model_loaded']}")
            print(f"Preprocessor loaded: {result['preprocessor_loaded']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the app is running with: python src/app.py")
        return False

if __name__ == "__main__":
    print("Testing Car Sales Prediction API...")
    print("=" * 50)

    # Test health endpoint
    print("\n1. Testing health check...")
    health_ok = test_health()

    if health_ok:
        # Test prediction with each example
        for i in range(len(examples)):
            print(f"\n2. Testing prediction with example {i+1}...")
            test_prediction(i)
    else:
        print("\n❌ Skipping prediction tests due to health check failure")

    print("\n" + "=" * 50)
    print("To run the web app: python src/app.py")
    print("Or use: run_api.bat (Windows)")
    print("Then visit: http://127.0.0.1:8000")