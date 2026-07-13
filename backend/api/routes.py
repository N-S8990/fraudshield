"""API routes for FraudShield."""
import sys
from pathlib import Path

# Ensure backend package is importable
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.predict import FraudPredictor

router = APIRouter(prefix="/api/v1")

# Transaction features that the model expects
class Transaction(BaseModel):
    time: float
    v1: float; v2: float; v3: float; v4: float; v5: float
    v6: float; v7: float; v8: float; v9: float; v10: float
    v11: float; v12: float; v13: float; v14: float; v15: float
    v16: float; v17: float; v18: float; v19: float; v20: float
    v21: float; v22: float; v23: float; v24: float; v25: float
    v26: float; v27: float; v28: float
    amount: float

    class Config:
        # Allow constructing from dict with all fields
        extra = "forbid"


class BatchRequest(BaseModel):
    transactions: list[Transaction]


predictor = FraudPredictor()


@router.post("/predict")
def predict_single(tx: Transaction):
    features = _tx_to_array(tx)
    result = predictor.predict(features)
    return {"fraud_probability": round(float(result["probability"]), 4),
            "prediction": int(result["prediction"]),
            "model": result["model"]}


@router.post("/predict/batch")
def predict_batch(req: BatchRequest):
    results = []
    for tx in req.transactions:
        features = _tx_to_array(tx)
        result = predictor.predict(features)
        results.append({"fraud_probability": round(float(result["probability"]), 4),
                        "prediction": int(result["prediction"])})
    return {"results": results, "model": predictor.loaded_model}


@router.get("/models")
def list_models():
    """Return which models are available and their metrics."""
    try:
        import json
        with open("models/metrics.json") as f:
            metrics = json.load(f)
        return {"models": metrics}
    except FileNotFoundError:
        return {"models": [], "note": "No trained models found. Run train.py first."}


def _tx_to_array(tx: Transaction) -> list:
    """Convert transaction pydantic model to feature list matching training order."""
    fields = ["time", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
              "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
              "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
              "v25", "v26", "v27", "v28", "amount"]
    return [getattr(tx, f) for f in fields]
