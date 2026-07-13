# FraudShield — Credit Card Fraud Detection

An end-to-end ML system for detecting fraudulent credit card transactions, with an interactive React dashboard.

Built by the Nirav Sayanja.

## Overview

Fraud detection is a critical problem in banking. This project trains three classification models (Logistic Regression, Random Forest, XGBoost) on the Kaggle Credit Card Fraud Detection dataset, handles severe class imbalance (0.17% fraud) with SMOTE, and serves predictions through a FastAPI backend with a live React dashboard.

## Tech Stack

| Layer | Technology |
|---|---|
| ML | Python, Scikit-learn, XGBoost, Pandas, SMOTE |
| Backend | FastAPI, uvicorn |
| Frontend | React 19, TypeScript, Vite, Recharts |
| Tracking | MLflow |
| Deployment | Docker (optional) |

## Project Structure

```
fraudshield/
├── backend/
│   ├── api/            # FastAPI app, routes, schemas
│   ├── models/         # Training script & prediction wrapper
│   └── requirements.txt
├── frontend/           # Vite + React + TypeScript dashboard
│   ├── src/
│   │   ├── pages/      # Dashboard, LivePredict
│   │   └── services/   # API client
│   └── package.json
├── notebooks/          # EDA and model experimentation
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_model_training.ipynb
├── models/             # Trained .pkl files (gitignored)
├── data/               # Datasets (gitignored)
└── docker-compose.yml  # (optional)
```

## Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python models/train.py              # Train models
PYTHONPATH=$PWD uvicorn api:app --reload
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — the dashboard loads model metrics and lets you run live predictions.

## Model Performance

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 5.8% | 91.8% | 0.109 | 0.970 |
| Random Forest | 83.5% | 82.7% | 0.831 | 0.964 |
| XGBoost | 68.3% | 85.7% | 0.760 | 0.975 |

**Default model**: XGBoost (best ROC-AUC, strong recall-precision tradeoff after threshold tuning).

## Dataset

Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraudulent (0.172%), 31 features (V1-V28 are PCA components, Time, Amount).

## Key Learnings

- Fraud detection requires handling **extreme class imbalance** — SMOTE and class weighting were essential
- **Simple models (Logistic Regression)** catch most frauds but drown operations in false positives
- **Ensemble methods (RF, XGBoost)** balance detection rate with operational cost
- This project covers the full ML pipeline from EDA to deployment with a real frontend

---

Built by Nirav Sayanja for the Nirav Sayanja.
