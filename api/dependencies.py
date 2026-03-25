from functools import lru_cache

from src.config import MODEL_PATH
from src.models.predict import load_model


@lru_cache(maxsize=1)
def get_model():
	"""Load model once and reuse it for all requests."""
	return load_model(MODEL_PATH)

