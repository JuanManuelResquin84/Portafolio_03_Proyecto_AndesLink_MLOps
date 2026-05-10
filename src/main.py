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

# Rutas de archivos (Caminos relativos desde src/ hasta models/)
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_churn_lrV2_andeslink.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler_andeslink.pkl')
COLS_PATH = os.path.join(BASE_DIR, '..', 'models', 'X_columns.pkl')
DNI_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_churn_lrV2_andeslink.json')

# Carga de artefactos
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
X_columns = joblib.load(COLS_PATH)

# Definición del Esquema de Entrada (Basado en tu entrenamiento real)
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

# Ruta de Inicio: Carga el DNI (Metadatos) del modelo
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
    # Convertir el objeto recibido a DataFrame de una fila
    df = pd.DataFrame([cliente.dict()])
    
    # Procesamiento de variables categóricas (get_dummies)
    # reindex asegura que las columnas sean exactamente las que el modelo espera
    df_proc = pd.get_dummies(df).reindex(columns=X_columns, fill_value=0)
    
    # Escalamiento de datos
    df_escalado = scaler.transform(df_proc)
    
    # Cálculo de probabilidad
    prob = float(model.predict_proba(df_escalado)[:, 1][0])
    
    # Aplicación del umbral óptimo definido en tu notebook (0.45)
    clase = 1 if prob >= 0.45 else 0
    
    return {
        "score_fuga": round(prob, 4),
        "umbral_aplicado": 0.45,
        "prediccion": "Fuga (1)" if clase == 1 else "Se queda (0)",
        "accion_recomendada": "Llamar para retención" if clase == 1 else "Mantener monitoreo"
    }
    
if __name__ == "__main__":
    import uvicorn
    # Arrancamos el servidor en el puerto 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)