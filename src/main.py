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
    version="2.0"
)

# Rutas de archivos
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_churn_GBC_andeslink.pkl')
DNI_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_churn_GBC_andeslink.json')

# Carga del único artefacto (Pipeline/Modelo unificado)
model = joblib.load(MODEL_PATH)

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

# Ruta de Inicio
@app.get("/")
def inicio():
    try:
        with open(DNI_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {"error": "Archivo de metadatos no encontrado."}
        
    return {
        "status": "Online",
        "proyecto": "AndesLink Churn Prediction",
        "metadata_modelo": metadata
    }

# Ruta de Predicción
@app.post("/predict")
def predecir(cliente: Cliente):
    # Convertir a DataFrame de una fila
    df = pd.DataFrame([cliente.model_dump()])
    
    # Predicción de clase usando el umbral genérico del modelo (0.5 por defecto)
    clase = int(model.predict(df)[0])
    
    # Cálculo de la probabilidad asociada
    prob = float(model.predict_proba(df)[:, 1][0])
    
    return {
        "score_fuga": round(prob, 4),
        "prediccion": "Fuga (1)" if clase == 1 else "Se queda (0)",
        "accion_recomendada": "Llamar para retención" if clase == 1 else "Mantener monitoreo"
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)