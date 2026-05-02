import joblib
import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import time
start_time = time.time()
clase, score = predecir_fuga(cliente_ejemplo)
end_time = time.time()

print(f"Tiempo real de predicción: {end_time - start_time:.4f} segundos")

def predecir_fuga(datos_cliente):
    try:
        
        # Detectamos dónde está este archivo app.py
        directorio_actual = os.path.dirname(__file__)
        
        # Construimos la ruta subiendo un nivel (..) para llegar a la raíz y luego entrar a 'models'
        # normpath limpia las barras inclinadas
        base_path = os.path.normpath(os.path.join(directorio_actual, '..', 'models'))
        
        # Definimos las rutas finales a los archivos
        modelo_path = os.path.join(base_path, 'modelo_churn_nbV1_andeslink.pkl')
        scaler_path = os.path.join(base_path, 'scaler_andeslink.pkl')
        cols_path = os.path.join(base_path, 'X_columns.pkl')

        # Carga de los elementos con las rutas
        modelo = joblib.load(modelo_path)
        scaler = joblib.load(scaler_path)
        columnas_maestras = joblib.load(cols_path)
        
        # Procesamiento
        df_nuevo = pd.DataFrame([datos_cliente])
        df_propio = pd.get_dummies(df_nuevo)
        
        # Alineación de columnas
        for col in columnas_maestras:
            if col not in df_propio.columns:
                df_propio[col] = 0
        
        df_propio = df_propio[columnas_maestras]
        
        # Escalar los datos
        datos_listos = scaler.transform(df_propio)
        
        # Predicción
        pred = modelo.predict(datos_listos)[0]
        proba = modelo.predict_proba(datos_listos)[0][1]
        
        return pred, proba

    except Exception as e:
        print(f"Error de predicción: {e}")
        return None, None

if __name__ == "__main__":
    # Datos según tu dataset original
    cliente_ejemplo = {
        'tenure_months': 7,
        'monthly_charge': 58.23,
        'total_charges': 326.50,
        'support_tickets': 2,
        'late_payments': 1,
        'avg_monthly_usage_gb': 81.83,
        'contract_type': 'mensual',
        'payment_method': 'transferencia',
        'internet_service': 'cable',
        'has_streaming': 0,
        'has_security_pack': 1,
        'num_products': 3,
        'region': 'centro',
        'customer_age': 53,
        'is_promo': 1
    }
    
    clase, score = predecir_fuga(cliente_ejemplo)
    
    if clase is not None:
        umbral_personalizado = 0.35
        resultado_ajustado = "ALERTA: Churn (Fuga)" if score >= umbral_personalizado else "Cliente Estable"
        
        print(f"\nResultado del Análisis (Umbral: {umbral_personalizado})")
        print(f"Estado: {resultado_ajustado}")
        print(f"Probabilidad de abandono: {score:.2%}")