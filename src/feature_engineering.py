import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV session data into a DataFrame.
    """
    return pd.read_csv(path, parse_dates=["login_time"])

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw session logs into numeric features.
    """
    # Extract hour of login
    df["login_hour"] = df["login_time"].dt.hour

    # Map categorical location into numbers
    df["location_code"] = df["location"].astype("category").cat.codes

    # Map categorical device into numbers
    df["device_code"] = df["device"].astype("category").cat.codes

    # Keep only feature columns
    features = df[["login_hour", "location_code", "device_code", "transaction_amount"]]
    return features

def generate_synthetic_data(n: int = 100) -> pd.DataFrame:
    """
    Generate synthetic normal user session data.
    """
    rng = np.random.default_rng()
    data = {
        "login_hour": rng.integers(8, 22, size=n),
        "location_code": rng.integers(0, 5, size=n),
        "device_code": rng.integers(0, 3, size=n),
        "transaction_amount": rng.integers(100, 20000, size=n)
    }
    return pd.DataFrame(data)
