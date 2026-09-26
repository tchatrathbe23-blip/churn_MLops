"""
Integration tests for the Churn Prediction FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "decision_threshold" in data
    assert "features" in data
    assert isinstance(data["features"], list)
    assert len(data["features"]) > 0


def test_ui_endpoint():
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Churn Prediction" in response.text


def test_root_redirect():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers.get("location") == "/ui"


def test_predict_success():
    sample_payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 358.2,
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk"] in ("high", "low")


def test_predict_invalid_data():
    invalid_payload = {
        "gender": "Female",
        # missing required fields
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
