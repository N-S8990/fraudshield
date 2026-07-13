"""
Prediction wrapper — loads the best model and runs inference.
"""
import json
from pathlib import Path

import joblib
import numpy as np


class FraudPredictor:
    def __init__(self, model_name: str = "xgboost"):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.metrics = None
        self._load()

    def _load(self):
        model_path = Path("models") / f"{self.model_name}.pkl"
        scaler_path = Path("models") / "scaler.pkl"
        metrics_path = Path("models") / "metrics.json"

        if model_path.exists():
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        if metrics_path.exists():
            with open(metrics_path) as f:
                self.metrics = json.load(f)

    def predict(self, features: list) -> dict:
        """Run prediction on raw feature list (order must match training)."""
        arr = np.array(features).reshape(1, -1)

        # Scale Time and Amount (indices 0 and -1)
        if self.scaler:
            arr[0, [0]] = self.scaler.transform(arr[:, [0]])  # Time
            arr[0, [-1]] = self.scaler.transform(arr[:, [-1]])  # Amount

        proba = self.model.predict_proba(arr)[0, 1]
        pred = int(self.model.predict(arr)[0])

        return {
            "prediction": pred,
            "probability": proba,
            "model": self.model_name,
        }

    @property
    def loaded_model(self):
        return self.model_name
