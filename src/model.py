# src/model.py

import joblib

class UEBAModel:
    """
    Wrapper class to load and predict using Isolation Forest model.
    """

    def __init__(self, model_path="model/model.pkl"):
        self.model = joblib.load(model_path)

    def predict_score(self, features):
        """
        Returns anomaly score (-ve means anomaly).
        """
        return self.model.decision_function(features)

    def predict_label(self, features):
        """
        Returns anomaly label: -1 (anomaly), 1 (normal).
        """
        return self.model.predict(features)
