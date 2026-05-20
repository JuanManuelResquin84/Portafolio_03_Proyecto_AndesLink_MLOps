import pandas as pd
import numpy as np
import os
import joblib

def preparar_datos():
    print("Pipeline de Preparación de Datos")
    
    # Rutas 
    BASE_DIR = os.path.dirname(__file__)
    INPUT_PATH = os.path.join(BASE_DIR, '..', 'data', 'churn_sintetico.csv')
    OUTPUT_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'churn_procesado.csv')
    OUTPUT_COLS_PATH = os.path.join(BASE_DIR, '..', 'models', 'X_columns.pkl')
    
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: No se encontró el archivo inicial en: {INPUT_PATH}")
        return

    # Carga del CSV Crudo
    df = pd.read_csv(INPUT_PATH)
    
    # Limpieza de Nulos 
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
        
    # Separación de características (X) y objetivo (y)
    X = df.drop(columns=['churn'])
    y = df['churn']

    # Aplicación de Variables Categóricas (get_dummies para 'contract_type')
    print("Aplicando One-Hot Encoding a las variables categóricas...")
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Guardar el orden exacto de las columnas estructuradas para la API
    print(f"Guardando la estructura de columnas en {OUTPUT_COLS_PATH}...")
    columns_list = X_encoded.columns.tolist()
    joblib.dump(columns_list, OUTPUT_COLS_PATH)
    
    # Reconstrucción del dataset final (X transformado + la columna churn)
    df_final = X_encoded.copy()
    df_final['churn'] = y
        
    # Exportar el CSV procesado listo para entrenar
    print(f"Exportando el dataset procesado a: {OUTPUT_DATA_PATH}")
    df_final.to_csv(OUTPUT_DATA_PATH, index=False)
    print("Proceso de preparación finalizado")

if __name__ == "__main__":
    preparar_datos()