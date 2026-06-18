import pandas as pd
import numpy as np

# Carga
ref_df = pd.read_csv("/app/data/churn_procesado.csv")
logs_df = pd.read_csv("/app/data/logs.csv")

# Convierte columnas binarias de vuelta a categóricas
def reverse_one_hot(df):
    # Esto es un ejemplo, adáptalo a tus columnas
    df = df.copy()
    if 'contract_type_mensual' in df.columns:
        df['contract_type'] = np.where(df['contract_type_mensual'] == True, 'mensual', 'anual')
    # repite para otras columnas
    return df

# Ahora ambos tienen la columna 'contract_type' como texto
ref_simple = reverse_one_hot(ref_df)
logs_simple = logs_df 

# Guardamos para Evidently
ref_simple.to_csv("/app/data/reference_prepared.csv", index=False)
logs_simple.to_csv("/app/data/current_prepared.csv", index=False)