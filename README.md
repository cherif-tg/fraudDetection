# Fraud Detection Project Starter

This repository is a production-style starter for a bank/payment fraud detection workflow.

## 1) Project Goal

Build an end-to-end fraud system with four phases:

- data analysis and feature engineering
- model training and evaluation
- API serving for real-time scoring
- monitoring and continuous improvements

## 2) Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements-dev.txt
```

3. Copy environment template:

```bash
copy .env.example .env
```

4. Run API:

```bash
uvicorn api.main:app --reload
```

5. Run tests:

```bash
pytest -q
```

## 3) Train Baseline Model

Place your CSV in `data/raw/` with at least these columns:

- `amount`
- `timestamp`
- `is_fraud`

Then train:

```bash
python -m src.models.train --input data/raw/transactions.csv --output models/model.joblib
```

## 4) API Endpoints

- `GET /health`
- `POST /predict`

Example payload:

```json
{
	"amount": 149.99,
	"timestamp": "2026-01-15T03:22:00Z",
	"customer_id": "C123",
	"merchant_id": "M456"
}
```

## 5) Next Steps

- replace baseline logistic regression with LightGBM/XGBoost
- add time-based validation and business cost metrics
- integrate SHAP reports and drift checks
- add MLflow experiment tracking and model registry
