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