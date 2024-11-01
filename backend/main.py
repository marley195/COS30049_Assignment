import logging
from typing import Dict
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import time
import numpy as np
import joblib
import tensorflow as tf
from pydantic import Field

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Global model variables
global regression_model, classification_model

# Load models with error handling
try:
    regression_model = joblib.load("model/regression_model.joblib")
    print("Regression model loaded successfully.")
except Exception as e:
    print(f"Error loading regression model: {e}")
    regression_model = None

try:
    classification_model = tf.keras.models.load_model("model/classification_model.keras")  # Adjusted to TensorFlow model loading
    print("Classification model loaded successfully.")
except Exception as e:
    print(f"Error loading classification model: {e}")
    classification_model = None

class AirQualityInput(BaseModel):
    Benzene: float
    CO: float
    NH3: float
    NO: float
    NO2: float
    NOx: float
    O3: float
    PM10: float
    PM2_5: float = Field(alias="PM2.5") # Adjusted to match the input field name
    SO2: float
    Toluene: float
    Xylene: float

aqc_map = {
    0: "Good",
    1: "Moderate",
    2: "Unhealthy",
    3: "Very Unhealthy",
    4: "Hazardous",
    5: "Severe"
}

@app.post("/predict")
def predict(input_data: AirQualityInput):
    if classification_model is None or regression_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded successfully.")
    # Concatenate all input tensors into a single tensor
    input_dict = {
        "Benzene": tf.constant([[float(input_data.Benzene)]], dtype=tf.float32),
        "CO": tf.constant([[float(input_data.CO)]], dtype=tf.float32),
        "NH3": tf.constant([[float(input_data.NH3)]], dtype=tf.float32),
        "NO": tf.constant([[float(input_data.NO)]], dtype=tf.float32),
        "NO2": tf.constant([[float(input_data.NO2)]], dtype=tf.float32),
        "NOx": tf.constant([[float(input_data.NOx)]], dtype=tf.float32),
        "O3": tf.constant([[float(input_data.O3)]], dtype=tf.float32),
        "PM10": tf.constant([[float(input_data.PM10)]], dtype=tf.float32),
        "PM2.5": tf.constant([[float(input_data.PM2_5)]], dtype=tf.float32),  # Ensure the key matches exactly
        "SO2": tf.constant([[float(input_data.SO2)]], dtype=tf.float32),
        "Toluene": tf.constant([[float(input_data.Toluene)]], dtype=tf.float32),
        "Xylene": tf.constant([[float(input_data.Xylene)]], dtype=tf.float32),
    }
    prediction = classification_model.predict(input_dict)
    print(f"Prediction number: {prediction}")
    predicted_class = np.argmax(prediction[0])
    rating_label = aqc_map.get(predicted_class, "Unknown")   
    # Prepare input as a single array for the regression model
    input_array = np.array(list(input_data.dict().values())).reshape(1, -1)
    print("Running regression model prediction with input array:", input_array)
    prediction = regression_model.predict(input_array)
    print("Prediction output from regression model:", prediction)
    rating = prediction[0]
    # Interpret the prediction
    #rating_label = interpret_rating(rating)
    return {rating, rating_label}


## Middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"request: {request.url} - Duration: {process_time} seconds")
    return response

## Exception handling for general errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error": "An error occurred"}
    )




"""
    this is for get functionailty when generating results from model.

    To avoid writing to disk we can stream the data back to the clien using the below code.
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="JPEG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/jpeg")

"""
