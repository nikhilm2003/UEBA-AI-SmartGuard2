# model/train_model.py

import os
import joblib
from sklearn.ensemble import IsolationForest
from src.feature_engineering import preprocess_data

def train_and_save_model(data_path="data/sample_logins.csv", model_path="model/model.pkl"):
    """
    Train an Isolation Forest model on login data.
    """
    X, df = preprocess_data(data_path)

    clf = IsolationForest(
        n_estimators=100,
        contamination=0.1,  # assume 10% anomalies
        random_state=42
    )
    clf.fit(X)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"✅ Model trained and saved to {model_path}")
    return clf

if __name__ == "__main__":
    train_and_save_model()
