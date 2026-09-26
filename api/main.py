"""
FastAPI serving for the trained churn model.

Loads the exported champion pipeline from a plain joblib file (artifacts/),
not from the MLflow registry. The model travels as a file baked into the image
instead -- see train.py's export step at the end of main() for where this file
comes from.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR.parent / "artifacts"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Churn Prediction API",
    description="Telecom customer churn prediction service with ML pipeline serving.",
    version="1.0.0",
)

# Static UI files
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Loaded once at startup, reused for every request -- not reloaded per call.
model = joblib.load(ARTIFACT_DIR / "champion_model.joblib")
meta = json.loads((ARTIFACT_DIR / "champion_meta.json").read_text())
THRESHOLD = meta["decision_threshold"]
FEATURES = meta["features"]  # exact column list + order the pipeline was trained on


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to UI."""
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
def ui():
    """Serve the web prediction interface."""
    index_file = STATIC_DIR / "index.html"
    return FileResponse(index_file)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    """Which model is actually live right now -- handy after every redeploy
    to confirm the new model made it, without guessing from the logs."""
    return meta


@app.post("/predict")
def predict(data: CustomerData):
    # FEATURES enforces the same column set/order the pipeline saw in
    # training; the ColumnTransformer selects by name either way, but this
    # also makes a missing/misnamed field fail loudly here instead of deep
    # inside the pipeline.
    row = pd.DataFrame([data.model_dump()])[FEATURES]
    prob = float(model.predict_proba(row)[0, 1])
    return {
        "churn_probability": round(prob, 4),
        "risk": "high" if prob >= THRESHOLD else "low",
    }
