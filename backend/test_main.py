from fastapi import FastAPI

app = FastAPI()

predictions_list = [[190, 32, 54, 123]]

@app.get("/predictions")
def get_predictions():
    return {"predictions": predictions_list}