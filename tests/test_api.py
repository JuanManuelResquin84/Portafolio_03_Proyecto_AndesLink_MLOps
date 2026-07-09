from fastapi.testclient import TestClient
import os
import sys

# Permitir que Python encuentre la carpeta src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import app

client = TestClient(app)


def test_inicio_api():
    """Prueba que el endpoint raíz funcione y devuelva el status Online"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Online"
    assert response.json()["proyecto"] == "AndesLink Churn Prediction"


def test_predict_input_valido():
    """Prueba que /predict devuelva una predicción correcta con datos válidos"""
    payload = {
        "tenure_months": 24,
        "monthly_charge": 75.5,
        "total_charges": 1800.0,
        "support_tickets": 1,
        "late_payments": 0,
        "avg_monthly_usage_gb": 100.0,
        "contract_type": "mensual",
        "payment_method": "tarjeta",
        "internet_service": "fibra",
        "has_streaming": 1,
        "has_security_pack": 0,
        "num_products": 2,
        "region": "centro",
        "customer_age": 35,
        "is_promo": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "score" in body
    assert "pred" in body
    assert body["pred"] in ["Fuga", "Se queda"]
    assert 0.0 <= body["score"] <= 1.0


def test_predict_edad_invalida():
    """Prueba que /predict rechace una edad fuera de rango (validación Field)"""
    payload = {
        "tenure_months": 24,
        "monthly_charge": 75.5,
        "total_charges": 1800.0,
        "support_tickets": 1,
        "late_payments": 0,
        "avg_monthly_usage_gb": 100.0,
        "contract_type": "mensual",
        "payment_method": "tarjeta",
        "internet_service": "fibra",
        "has_streaming": 1,
        "has_security_pack": 0,
        "num_products": 2,
        "region": "centro",
        "customer_age": 300,  # inválido: supera le=120
        "is_promo": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_tenure_negativo():
    """Prueba que /predict rechace tenure_months negativo (validación Field)"""
    payload = {
        "tenure_months": -5,  # inválido: menor a ge=0
        "monthly_charge": 75.5,
        "total_charges": 1800.0,
        "support_tickets": 1,
        "late_payments": 0,
        "avg_monthly_usage_gb": 100.0,
        "contract_type": "mensual",
        "payment_method": "tarjeta",
        "internet_service": "fibra",
        "has_streaming": 1,
        "has_security_pack": 0,
        "num_products": 2,
        "region": "centro",
        "customer_age": 35,
        "is_promo": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_campo_faltante():
    """Prueba que /predict rechace un request incompleto"""
    payload = {"tenure_months": 24}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint():
    """Prueba que /metrics exponga el formato Prometheus"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "ml_predictions_total" in response.text
    assert "ml_model_accuracy" in response.text