from fastapi import FastAPI
import joblib
import pandas as pd
import os
import json
from pydantic import BaseModel
import time #para medir latencia

# IMPORTAR PROMETHEUS CLIENT 
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

# Inicializar la App
app = FastAPI(
    title="AndesLink Churn API", 
    description="API de predicción de fuga de clientes para el Proyecto AndesLink",
    version="1.0"  
)

# DEFINIR LAS MÉTRICAS DE MONITOREO 
# Técnico: Monitorea cuánto tarda el endpoint /predict
LATENCIA_PREDICCION = Histogram(
    "ml_prediction_latency_seconds", 
    "Tiempo de respuesta del endpoint de prediccion en segundos"
)

# Negocio/Modelo: Cuenta cuántas peticiones ingresan y la decisión tomada
CONTEO_PREDICCIONES = Counter(
    "ml_predictions_total", 
    "Total de predicciones realizadas por la API",
    ["resultado_prediccion", "model_version"] # Etiquetas para agrupar en Grafana
)

# Riesgo/Datos: Registra el último score de fuga calculado para ver si el mercado cambia
ULTIMO_SCORE_FUGA = Gauge(
    "ml_last_churn_score", 
    "Ultima probabilidad de fuga calculada por el modelo"
)


# Rutas de archivos 
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_churn_GBC_andeslink.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler_andeslink.pkl')      
METRICS_PATH = os.path.join(BASE_DIR, '..', 'models', 'metrics.json')             
COLS_PATH = os.path.join(BASE_DIR, '..', 'models', 'X_columns.pkl') # Ruta a las columnas del entrenamiento

# Carga de los artefactos necesarios
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)  # Cargamos el escalador que guardó train.py
model_columns = joblib.load(COLS_PATH) # Cargamos la lista de columnas de prepare.py

# Definición del Esquema de Entrada 
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

# Ruta de Inicio (Muestra las métricas locales reales, incluyendo tu Kappa)
@app.get("/")
def inicio():
    try:
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {"error": "Archivo metrics.json no encontrado."}
        
    return {
        "status": "Online",
        "proyecto": "AndesLink Churn Prediction",
        "metadata_modelo_actual": metadata
    }

# Ruta de Predicción ajustada al proceso manual e instrumentada con Prometheus
@app.post("/predict")
def predecir(cliente: Cliente):
    # SEÑAL TÉCNICA: Iniciar reloj para medir latencia
    inicio_tiempo = time.time()
    
    # Convertir a DataFrame de una fila
    df = pd.DataFrame([cliente.model_dump()])
    
    # === PROCESAMIENTO CATEGÓRICO EN API ===
    # 1. Dummificamos normal para que no borre la única fila ingresada
    df_encoded = pd.get_dummies(df)
    
    # 2. Reindexamos usando la lista exacta del entrenamiento (model_columns).
    # Al mapear contra la lista que ya tiene el 'drop_first' heredado de prepare.py,
    # la estructura se alinea perfectamente de forma matemática. Las columnas faltantes van con 0.
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
    
    # ESCALADO REQUERIDO: Ahora el DataFrame es idéntico al que espera el Scaler
    datos_escalados = scaler.transform(df_final)
    
    # CÁLCULO DE PROBABILIDAD: Obtenemos el score de fuga real
    probabilidad = float(model.predict_proba(datos_escalados)[:, 1][0])
    
    # REGLA DE NEGOCIO EFECTIVA: Aplicamos tu umbral personalizado de 0.45
    umbral = 0.45
    clase = 1 if probabilidad >= umbral else 0
    
    texto_prediccion = "Fuga (1)" if clase == 1 else "Se queda (0)"
    
    # REGISTRAR SEÑALES EN PROMETHEUS
    ULTIMO_SCORE_FUGA.set(probabilidad)
    CONTEO_PREDICCIONES.labels(resultado_prediccion=texto_prediccion, model_version="GBC_V1").inc()
    
    # Calcula la latencia final y la guarda en el histograma
    latencia = time.time() - inicio_tiempo
    LATENCIA_PREDICCION.observe(latencia)
    
    return {
        "score_fuga": round(probabilidad, 4),
        "umbral_aplicado": umbral,
        "prediccion": texto_prediccion,
        "accion_recomendada": "Llamar para retención inmediata (Área de Fidelización)" if clase == 1 else "Mantener monitoreo estándar"
    }

# EXPONER EL ENDPOINT /metrics (Observabilidad)
app.mount("/metrics", make_asgi_app())
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)