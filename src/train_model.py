import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.feature_engineering import load_data, preprocess, generate_synthetic_data

MODEL_PATH = "model/model.pkl"

def train():
    """
    Train Isolation Forest on session data.
    """
    # Load sample data
    try:
        df = load_data("data/sample_sessions.csv")
        X = preprocess(df)
    except Exception:
        print("No data found, generating synthetic data...")
        X = generate_synthetic_data(200)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_scaled)

    # Ensure model directory exists
    os.makedirs("model", exist_ok=True)

    # Save model and scaler
    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train()
