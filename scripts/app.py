import joblib
import pandas as pd

# Ruta al modelo guardado
MODEL_PATH = 'models/modelo_churn_nbV1_andeslink.pkl'

def cargar_modelo():
    try:
        modelo = joblib.load(MODEL_PATH)
        print("Modelo cargado exitosamente.")
        return modelo
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def predecir_nuevo_cliente(datos_cliente):
    
    # datos_cliente: debe ser un DataFrame con las mismas columnas que X_train
    modelo = cargar_modelo()
    if modelo:
        prediccion = modelo.predict(datos_cliente)
        probabilidad = modelo.predict_proba(datos_cliente)
        return prediccion, probabilidad
    return None, None

if __name__ == "__main__":
    # Esto es solo una prueba para verificar l funcionalidad
    print("Ejecutando prueba de inferencia.")
    
    # Aquí deberías crear un ejemplo con datos ficticios que sigan el formato de tu X_test
    # Ejemplo: df_prueba = pd.DataFrame([[...]], columns=['col1', 'col2', ...])
    # prediccion, proba = predecir_nuevo_cliente(df_prueba)
    # print(f"Predicción: {prediccion}")