from fastapi import FastAPI
import joblib
import pandas as pd
import os
import json
from pydantic import BaseModel

# Inicializar la App
app = FastAPI(
    title="AndesLink Churn API", 
    description="API de predicción de fuga de clientes para el Proyecto AndesLink",
    version="1.0"  
)

# Rutas de archivos 
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_churn_GBC_andeslink.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler_andeslink.pkl')      
METRICS_PATH = os.path.join(BASE_DIR, '..', 'models', 'metrics.json')             

# Carga de los artefactos necesarios
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)  #Cargamos el escalador que guardó train.py

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

# Ruta de Predicción ajustada al proceso manual
@app.post("/predict")
def predecir(cliente: Cliente):
    # 1. Convertir a DataFrame de una fila
    df = pd.DataFrame([cliente.model_dump()])
    
    # ESCALADO REQUERIDO: Pasamos los datos por el transform del Scaler entrenado
    datos_escalados = scaler.transform(df)
    
    # CÁLCULO DE PROBABILIDAD: Obtenemos el score de fuga real
    probabilidad = float(model.predict_proba(datos_escalados)[:, 1][0])
    
    # REGLA DE NEGOCIO EFECTIVA: Aplicamos tu umbral personalizado de 0.45
    umbral = 0.45
    clase = 1 if probabilidad >= umbral else 0
    
    return {
        "score_fuga": round(probabilidad, 4),
        "umbral_aplicado": umbral,
        "prediccion": "Fuga (1)" if clase == 1 else "Se queda (0)",
        "accion_recomendada": "Llamar para retención inmediata (Área de Fidelización)" if clase == 1 else "Mantener monitoreo estándar"
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)