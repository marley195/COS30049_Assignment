from fastapi.testclient import TestClient
import numpy as np
from main import app, AirQualityInput  # Make sure to import your FastAPI app and data model

# Instantiate the TestClient with the app
client = TestClient(app)

# Mock input data and model selection
mock_data = {
    "Benzene": 2.5,
    "CO": 0.8,
    "NH3": 4.3,
    "NO": 1.1,
    "NO2": 15.6,
    "NOx": 18.2,
    "O3": 30.4,
    "PM10": 20.0,
    "PM2_5": 12.5,  # Note: Using "PM2_5" to match the model field name
    "SO2": 5.0,
    "Toluene": 3.2,
    "Xylene": 1.8
}
model_choice = "Classification"  # Change to "Regression" if you want to test the regression model

def test_predict_endpoint():
    # Make a POST request to the /predict endpoint
    response = client.post(f"/predict?model_choice={model_choice}", json=mock_data)
    
    # Check if the response is successful
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Print or inspect the response JSON
    result = response.json()
    print("Prediction Result:", result)
    
    # Check that the response contains expected keys
    assert "rating" in result, "Response missing 'rating'"
    assert "rating_label" in result, "Response missing 'rating_label'"
    
    # Optional: Add assertions to verify expected ranges or labels
    # e.g., Check if rating is a float and rating_label is a string
    assert isinstance(result["rating"], (int, float)), "Rating is not a number"
    assert isinstance(result["rating_label"], str), "Rating label is not a string"

# Run the test function
test_predict_endpoint()
