from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

TARGET_COLUMN = "is_fraud"
TIMESTAMP_COLUMN = "timestamp"

APP_NAME = os.getenv("APP_NAME", "Fraud Detection API")
APP_ENV = os.getenv("APP_ENV", "dev")
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.joblib"))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.50"))

