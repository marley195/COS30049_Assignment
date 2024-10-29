from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks
import io
from fastapi.responses import StreamingResponse
import time
import pickle
import numpy as np
import tensorflow as tf
##NEED AT LEAST 4 

app = FastAPI()

model = None

#Load in first trained model
classifcation_model = tf.keras.models.load_model("model/classification_model.keras")
#Load in second trained model
with open("model/regression_model.pkl", "rb") as reg_model_file:
    regression_model = pickle.load(reg_model_file)


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
class AirQualityInput(BaseModel):
    PM25 : float
    PM10 : float
    co : float
    SO2 : float
    NO2 : float
    O3 : float
    model_choice: str

##Define the post function for the predict endpoint.
@app.post("/predict")
async def predict(input_data: AirQualityInput):
    input_array = np.array([[input_data.PM25, input_data.PM10, input_data.co, input_data.SO2, input_data.NO2, input_data.O3]])
    
    if input.model == "Classifcation":
        selected_model = classifcation_model
    elif input_data.model == "Regression":
        selected_model = regression_model
    else:
        raise HTTPException(status_code=400, detail="Invalid model choice. Please choose Classifcation or Regression")
    
    try:
        prediction = selected_model.predict(input_array)
        rating = prediction[0]

        rating_label = interpret_rating(rating)

        return {
            "rating": rating, 
            "rating_label": rating_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Predection failed") from e
    

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




"""
    this is for get functionailty when generating results from model.

    To avoid writing to disk we can stream the data back to the clien using the below code.
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="JPEG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/jpeg")

"""