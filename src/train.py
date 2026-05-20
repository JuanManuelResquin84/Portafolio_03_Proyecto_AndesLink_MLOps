import pandas as pd
import numpy as np
import os
import joblib
import json  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score, cohen_kappa_score 
import mlflow

def entrenar_modelo():
    print("Iniciando Pipeline de Entrenamiento")
    
    # Rutas
    BASE_DIR = os.path.dirname(__file__)
    INPUT_PATH = os.path.join(BASE_DIR, '..', 'data', 'churn_procesado.csv')
    MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'models', 'modelo_churn_GBC_andeslink.pkl')
    SCALER_OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler_andeslink.pkl')
    METRICS_OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'models','metrics.json')
    
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: No se encuentra el dataset procesado en: {INPUT_PATH}. Corré primero src/prepare.py")
        return

    # Carga de datos
    df = pd.read_csv(INPUT_PATH)
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)

    # Configuración de MLflow
    with mlflow.start_run(run_name="Gradient_Boosting_Train"):
        params = {

            "n_estimators": 100,

            "learning_rate": 0.1,

            "max_depth": 3,

            "random_state": 42

        }
        
        model = GradientBoostingClassifier(**params)
        model.fit(X_train_scaled, y_train)
        
        # Predicción con tu umbral de negocio (0.45)
        probabilidades = model.predict_proba(X_test_scaled)[:, 1]
        umbral = 0.45
        y_pred = (probabilidades >= umbral).astype(int)
        
        # Cálculo de Métricas
        metricas = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "kappa": float(cohen_kappa_score(y_test, y_pred))
        }
        
        print(f"Métricas -> Recall: {metricas['recall']:.4f} | F1: {metricas['f1_score']:.4f} | Kappa: {metricas['kappa']:.4f}")
        
        # Registro en MLflow
        mlflow.log_params(params)
        mlflow.log_param("umbral_personalizado", umbral)
        mlflow.log_metrics(metricas)
        
        joblib.dump(model, MODEL_OUTPUT_PATH)
        mlflow.log_artifact(MODEL_OUTPUT_PATH)
        
        # GUARDAR EL ARCHIVO JSON LOCAL 
        print(f"Guardando métricas locales en: {METRICS_OUTPUT_PATH}...")
        with open(METRICS_OUTPUT_PATH, "w") as f:
            json.dump(metricas, f, indent=2)
        
    print("Proceso de entrenamiento finalizado")

if __name__ == "__main__":
    entrenar_modelo()