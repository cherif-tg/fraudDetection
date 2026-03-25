from fastapi import FastAPI

from api.routers.health import router as health_router
from api.routers.predict import router as predict_router
from src.config import APP_NAME


app = FastAPI(title=APP_NAME, version="0.1.0")
app.include_router(health_router)
app.include_router(predict_router)

