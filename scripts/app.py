import joblib
import pandas as pd
import numpy as np

def predecir_fuga(datos_cliente_dict):
    try:
        # Carga de los elementos
        modelo = joblib.load('models/modelo_churn_nbV1_andeslink.pkl')
        scaler = joblib.load('models/scaler_andeslink.pkl')
        columnas_maestras = joblib.load('models/X_columns.pkl')
        
        # Convertir el diccionario del cliente a DataFrame
        df_nuevo = pd.DataFrame([datos_cliente_dict])
        
        # Convertir a dummies (categorías a números)
        df_propio = pd.get_dummies(df_nuevo)
        
        # Alineación de columnas:
        # Esto rellena con 0 las columnas que el modelo espera pero el cliente no tiene
        for col in columnas_maestras:
            if col not in df_propio.columns:
                df_propio[col] = 0
        
        # Asegurar el orden exacto de las columnas
        df_propio = df_propio[columnas_maestras]
        
        # Escalar los datos
        datos_listos = scaler.transform(df_propio)
        
        # Predicción
        pred = modelo.predict(datos_listos)[0]
        proba = modelo.predict_proba(datos_listos)[0][1]
        
        return pred, proba

    except Exception as e:
        print(f"Error en la predicción: {e}")
        return None, None

if __name__ == "__main__":
    # Ejemplo de un cliente ficticio para probar
    cliente_ejemplo = {
        'antiguedad_meses': 5,
        'monto_mensual': 8500,
        'tipo_plan': 'Fibra 100MB',
        'metodo_pago': 'Efectivo',
        'reclamos_mes': 3
    }
    
    clase, score = predecir_fuga(cliente_ejemplo)
    
    if clase is not None:
        resultado = "ALERTA: Churn (Fuga)" if clase == 1 else "Cliente Estable"
        print(f"\n--- Resultado del Análisis ---")
        print(f"Estado: {resultado}")
        print(f"Probabilidad de abandono: {score:.2%}")
    