import joblib
import numpy as np
import pandas as pd
from src.feature_engineering import preprocess

MODEL_PATH = "model/model.pkl"

def load_model():
    """
    Load trained model and scaler from disk.
    """
    data = joblib.load(MODEL_PATH)
    return data["model"], data["scaler"]

def predict(session: dict) -> dict:
    """
    Predict anomaly score for a new session.
    session = {
        "login_time": "2025-09-29 03:00:00",
        "location": "Russia",
        "device": "Android",
        "transaction_amount": 500000
    }
    """
    model, scaler = load_model()

    # Convert to DataFrame
    df = pd.DataFrame([session])
    df["login_time"] = pd.to_datetime(df["login_time"])
    X = preprocess(df)

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    score = model.decision_function(X_scaled)[0]
    pred = model.predict(X_scaled)[0]  # -1 = anomaly, 1 = normal

    return {
        "score": float(score),
        "prediction": "Anomaly" if pred == -1 else "Normal"
    }
