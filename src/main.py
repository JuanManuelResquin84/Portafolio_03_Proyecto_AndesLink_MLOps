import os
import time
import csv
import logging
import joblib
import pandas as pd
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pydantic import BaseModel

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Métricas
LATENCIA = Histogram("ml_prediction_latency_seconds", "Latencia", buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5))
CONTEO = Counter("ml_predictions_total", "Total", ["resultado", "version"])
ACCURACY = Gauge("ml_model_accuracy", "Accuracy")

# Carga de modelos y datos
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, '../models/modelo_churn_GBC_andeslink.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, '../models/scaler_andeslink.pkl'))
cols = joblib.load(os.path.join(BASE_DIR, '../models/X_columns.pkl'))
ACCURACY.set(0.8421)

class Cliente(BaseModel):
    tenure_months: int
    monthly_charge: float
    total_charges: float
    support_tickets: int
    late_payments: int
    avg_monthly_usage_gb: float
    contract_type: str
    payment_method: str
    internet_service: str
    has_streaming: int
    has_security_pack: int
    num_products: int
    region: str
    customer_age: int
    is_promo: int

def registrar_log(data: dict):
    path = os.path.join(BASE_DIR, '../data/logs.csv')
    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(data)

# --- RUTAS ---

@app.get("/")
def read_root():
    """Endpoint para verificar el estado de la API"""
    return {"status": "Online"}

@app.post("/predict")
def predecir(c: Cliente, bt: BackgroundTasks):
    ini = time.time()
    df = pd.DataFrame([c.model_dump()])
    df_f = pd.get_dummies(df).reindex(columns=cols, fill_value=0)
    prob = float(model.predict_proba(scaler.transform(df_f))[:, 1][0])
    res = "Fuga" if prob >= 0.45 else "Se queda"
    CONTEO.labels(resultado=res, version="V1").inc()
    LATENCIA.observe(time.time() - ini)
    bt.add_task(registrar_log, c.model_dump())
    return {"score": round(prob, 4), "pred": res}

@app.get("/metrics")
def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)