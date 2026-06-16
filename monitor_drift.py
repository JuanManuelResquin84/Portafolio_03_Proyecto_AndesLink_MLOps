import os
import time
import pandas as pd
from evidently.report import Report
from evidently.presets import DataDriftPreset

# Rutas dentro del contenedor
RUTA_DATOS = "/app/data/churn_procesado.csv" 
CARPETA_REPORTES = "/app/data" 
INTERVALO_SEGUNDOS = 3600  # 1 hora

def calcular_y_guardar_drift():
    print(f"\n[{time.strftime('%H:%M:%S')}] Iniciando análisis de drift...")
    
    if not os.path.exists(RUTA_DATOS):
        print(f"Error: No se encontró el archivo en {RUTA_DATOS}")
        return False
    
    # Cargar datos
    data_ref = pd.read_csv(RUTA_DATOS)
    data_cur = data_ref.copy()
    
    # Simular un poco de drift para que el reporte sea interesante
    data_cur['monthly_charge'] = data_cur['monthly_charge'] * 1.25
    
    # Generar Reporte
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=data_ref, current_data=data_cur)
    
    # Guardar en la carpeta que el servidor de Docker va a publicar
    nombre_archivo = "index.html"
    ruta_completa = os.path.join(CARPETA_REPORTES, nombre_archivo)
    
    try:
        drift_report.save_html(ruta_completa)
        print(f"Reporte actualizado exitosamente en: {ruta_completa}")
        return True
    except Exception as e:
        print(f"Error al guardar: {e}")
        return False

if __name__ == "__main__":
    # Asegurar que la carpeta exista
    os.makedirs(CARPETA_REPORTES, exist_ok=True)
    
    print("Servicio de monitoreo iniciado.")
    
    while True:
        calcular_y_guardar_drift()
        print(f"Esperando {INTERVALO_SEGUNDOS} segundos para el próximo análisis...")
        time.sleep(INTERVALO_SEGUNDOS)