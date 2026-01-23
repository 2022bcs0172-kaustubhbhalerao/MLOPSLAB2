from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.joblib")

@app.post("/predict")
def predict(features: list):
    data = np.array(features).reshape(1, -1)
    prediction = int(model.predict(data)[0])
    return {
        "name": "Kaustubh Bhalerao",
        "roll_no": "2022BCS0172",
        "wine_quality": prediction
    }
