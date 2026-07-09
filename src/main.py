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
from pydantic import BaseModel, Field

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Métricas
LATENCIA = Histogram("ml_prediction_latency_seconds", "Latencia", buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5))
CONTEO = Counter("ml_predictions_total", "Total", ["resultado", "version"])
ACCURACY = Gauge("ml_model_accuracy", "Accuracy")

# Rutas
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '../models/modelo_churn_GBC_andeslink.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '../models/scaler_andeslink.pkl')
COLS_PATH = os.path.join(BASE_DIR, '../models/X_columns.pkl')
METRICS_PATH = os.path.join(BASE_DIR, '../models/metrics.json')

# Carga de modelos y datos
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    cols = joblib.load(COLS_PATH)
except FileNotFoundError as e:
    logger.error(f"No se pudo cargar un artefacto del modelo: {e}")
    raise RuntimeError(
        "Faltan artefactos del modelo. Corré prepare.py y train.py antes de levantar la API."
    ) from e

# Accuracy real desde metrics.json (generado por train.py), en vez de un valor hardcodeado
try:
    with open(METRICS_PATH) as f:
        metricas = json.load(f)
    ACCURACY.set(metricas.get("accuracy", 0.0))
    logger.info(f"Accuracy cargada desde metrics.json: {metricas.get('accuracy')}")
except FileNotFoundError:
    logger.warning("metrics.json no encontrado. ACCURACY inicializada en 0. Corré train.py primero.")
    ACCURACY.set(0.0)
except (json.JSONDecodeError, KeyError) as e:
    logger.warning(f"metrics.json inválido: {e}. ACCURACY inicializada en 0.")
    ACCURACY.set(0.0)


class Cliente(BaseModel):
    tenure_months: int = Field(ge=0, le=600, description="Antigüedad en meses")
    monthly_charge: float = Field(gt=0, description="Cargo mensual, debe ser positivo")
    total_charges: float = Field(ge=0)
    support_tickets: int = Field(ge=0)
    late_payments: int = Field(ge=0)
    avg_monthly_usage_gb: float = Field(ge=0)
    contract_type: str
    payment_method: str
    internet_service: str
    has_streaming: int = Field(ge=0, le=1)
    has_security_pack: int = Field(ge=0, le=1)
    num_products: int = Field(ge=0)
    region: str
    customer_age: int = Field(ge=0, le=120)
    is_promo: int = Field(ge=0, le=1)


def registrar_log(data: dict):
    path = os.path.join(BASE_DIR, '../data/logs.csv')
    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


# --- RUTAS ---

@app.get("/")
def read_root():
    """Endpoint para verificar el estado de la API"""
    return {
        "status": "Online",
        "proyecto": "AndesLink Churn Prediction"
    }


@app.post("/predict")
def predecir(c: Cliente, bt: BackgroundTasks):
    ini = time.time()
    try:
        df = pd.DataFrame([c.model_dump()])
        df_f = pd.get_dummies(df).reindex(columns=cols, fill_value=0)
        prob = float(model.predict_proba(scaler.transform(df_f))[:, 1][0])
    except Exception as e:
        logger.error(f"Error al predecir: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la predicción.")

    res = "Fuga" if prob >= 0.45 else "Se queda"
    CONTEO.labels(resultado=res, version="V1").inc()
    LATENCIA.observe(time.time() - ini)
    bt.add_task(registrar_log, c.model_dump())
    return {"score": round(prob, 4), "pred": res}


@app.get("/metrics")
def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
