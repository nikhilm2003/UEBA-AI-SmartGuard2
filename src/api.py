# src/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from src.model import UEBAModel

app = FastAPI(title="AI-Powered UEBA API")

# Load trained model
ueba_model = UEBAModel(model_path="model/model.pkl")

# Request format
class LoginEvent(BaseModel):
    login_hour: int
    location_id: int
    device_id: int

@app.post("/predict")
def predict(event: LoginEvent):
    features = np.array([[event.login_hour, event.location_id, event.device_id]])
    score = float(ueba_model.predict_score(features))
    label = int(ueba_model.predict_label(features)[0])

    return {
        "anomaly_score": score,
        "prediction": "anomaly" if label == -1 else "normal"
    }
