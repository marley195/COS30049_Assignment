from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks
import io
from fastapi.responses import StreamingResponse
import time
import pickle
import numpy as np
import tensorflow as tf
import joblib
##NEED AT LEAST 4 

app = FastAPI()


prediction_history = []
model = None

from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def load_models():
    global regression_model, classification_model

    try:
        regression_model = joblib.load("model/regression_model.joblib")
        print("Regression model loaded successfully.")
    except Exception as e:
        print(f"Error loading regression model: {e}")
        regression_model = None

    try:
        classification_model = joblib.load("model/classification_model.joblib")
        print("Classification model loaded successfully.")
    except Exception as e:
        print(f"Error loading classification model: {e}")
        classification_model = None


"""
    Define data model for the input using pydantic.
    Main Metrics:
    Particle Matter2.5 (PM.25)
    Pariculate Matter10 (PM.10)
    carbon monoxide (co)
    sulphur dioxide (SO2)
    nitrogen dioxide (NO2)
    ozone (O3)
"""

predictions = []

class AirQualityInput(BaseModel):
    PM25 : float
    PM10 : float
    co : float
    SO2 : float
    NO2 : float
    O3 : float


##Define the post function for the predict endpoint.
@app.post("/predict")
def predict(input_data: AirQualityInput, model_choice: str = Query(...)):
    # Convert input_data to a numpy array
    input_array = np.array(list(dict(input_data).values())).reshape(1, -1)
    
    # Select the appropriate model
    if model_choice == "Classification":
        selected_model = classification_model
    elif model_choice == "Regression":
        selected_model = regression_model
    else:
        raise HTTPException(status_code=400, detail="Invalid model choice. Please choose 'Classification' or 'Regression'")
    
    # Make a prediction
    try:
        prediction = selected_model.predict(input_array)
        rating = prediction[0]

        # Interpret the rating
        rating_label = interpret_rating(rating)

        return {"rating": 85, "rating_label": "Moderate"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed") from e
    

##Basic function for interpreting the rating !!MAKE MORE EXTENSIVE!!
def interpret_rating(rating):
    if rating == 0:
        return "Good"
    elif rating == 1:
        return "Satisfactory"
    elif rating == 2:
        return "Moderate"
    elif rating == 3:
        return "Poor"
    elif rating == 4:
        return "Very Poor"
    elif rating == 5:
        return "Severe"
    else:
        return "Invalid Rating"

@app.get("/prediction-history/")
async def get_prediction_history():
    return prediction_history


##Middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"request: {request.url} - Duration: {process_time} seconds")
    return response
##Exception handling for HTTP errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )
##Exception handling for general errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail,
        "error": "An error occurred"})


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)

"""
    this is for get functionailty when generating results from model.

    To avoid writing to disk we can stream the data back to the clien using the below code.
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="JPEG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/jpeg")

"""
