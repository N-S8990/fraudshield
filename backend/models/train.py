"""
Model training script for credit card fraud detection.

Trains Logistic Regression, Random Forest, and XGBoost classifiers
on the Kaggle Credit Card Fraud Detection dataset with SMOTE oversampling.
Logs experiments via MLflow.
"""
import json
import os
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------- paths ----------
DATA_PATH = Path("data/raw/creditcard.csv")
MODELS_DIR = Path("models")
METRICS_PATH = MODELS_DIR / "metrics.json"
EXPERIMENT_NAME = "fraudshield"

os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_prep_data():
    """Load the Kaggle dataset, split, and scale."""
    df = pd.read_csv(DATA_PATH)

    # Quick sanity
    assert "Class" in df.columns, "Dataset must have a 'Class' column"
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} / {len(df)} ({df['Class'].mean()*100:.4f}%)")

    # Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Train/test split (stratified to keep fraud ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale Amount and Time (V1-V28 are already PCA-scaled)
    scaler = StandardScaler()
    for col in ["Time", "Amount"]:
        X_train[col] = scaler.fit_transform(X_train[[col]])
        X_test[col] = scaler.transform(X_test[[col]])

    # Apply SMOTE to training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE — train size: {X_train_res.shape[0]}, fraud ratio: {y_train_res.mean()*100:.2f}%")
    return X_train_res, X_test, y_train_res, y_test, scaler


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """Train a model, log metrics, save artifact. Returns metrics dict."""
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "model": model_name,
        }

        # Log to MLflow
        mlflow.log_params(
            {"model_type": model_name, "smote": True}
        )
        mlflow.log_metrics(
            {k: float(v) for k, v in metrics.items() if k != "model"}
        )
        # Trust xgboost types for MLflow serialization
        kwargs = {}
        if "xgboost" in str(type(model)):
            kwargs["skops_trusted_types"] = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
        mlflow.sklearn.log_model(model, model_name, **kwargs)

        print(f"\n--- {model_name} ---")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        # Save model
        path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(model, path)
        print(f"  Saved → {path}")

        return metrics


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("=" * 50)
    print("FraudShield — Model Training")
    print("=" * 50)

    X_train, X_test, y_train, y_test, scaler = load_and_prep_data()

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
    }

    all_metrics = []
    for name, model in models.items():
        m = train_and_evaluate(model, name, X_train, X_test, y_train, y_test)
        all_metrics.append(m)

    # Save metrics for the API
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved → {METRICS_PATH}")

    # Save scaler for inference
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"Scaler saved → {MODELS_DIR / 'scaler.pkl'}")

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
