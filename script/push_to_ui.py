import os
import time
import pandas as pd
import joblib
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.ui.workspace import RemoteWorkspace

REF_PATH = "/app/data/churn_procesado.csv"
LOGS_PATH = "/app/data/logs.csv"
COLS_PATH = "/app/models/X_columns.pkl"
EVIDENTLY_URL = "http://evidently:8000"
PROJECT_NAME = "Monitoreo_AndesLink"

MIN_LOGS = 4              # mínimo de registros de producción para que el drift tenga sentido
INTERVALO_SEGUNDOS = 30  # cada 30 segundos (ajustable: 6-24hs según lo documentado en el README)


def generar_y_subir_reporte():
    print(f"\n[{time.strftime('%H:%M:%S')}] Iniciando ciclo de monitoreo...")

    if not os.path.exists(LOGS_PATH):
        print("Aún no existe logs.csv. Esperando predicciones de la API.")
        return False

    if not os.path.exists(REF_PATH):
        print(f"No se encontró el archivo de referencia: {REF_PATH}")
        return False

    if not os.path.exists(COLS_PATH):
        print(f"No se encontró el archivo de columnas: {COLS_PATH}")
        return False

    logs = pd.read_csv(LOGS_PATH)
    if len(logs) < MIN_LOGS:
        print(f"Solo hay {len(logs)} logs, se necesitan al menos {MIN_LOGS}. Se omite este ciclo.")
        return False

    try:
        ref = pd.read_csv(REF_PATH).drop(columns=["churn"], errors="ignore")
        cols = joblib.load(COLS_PATH)

        logs_transformed = pd.get_dummies(logs).reindex(columns=cols, fill_value=0)

        ref_dataset = Dataset.from_pandas(ref, data_definition=DataDefinition())
        current_dataset = Dataset.from_pandas(logs_transformed, data_definition=DataDefinition())

        report = Report([DataSummaryPreset(), DataDriftPreset()])
        run = report.run(current_dataset, ref_dataset)

        ws = RemoteWorkspace(EVIDENTLY_URL)
        project = ws.create_project(PROJECT_NAME)
        ws.add_run(project.id, run)

        print(f"Reporte subido correctamente con {len(logs)} registros de producción.")
        print(f"Ver en: http://localhost:8001")
        return True

    except Exception as e:
        print(f"Error al generar/subir el reporte: {e}")
        return False


if __name__ == "__main__":
    print("Servicio de monitoreo Evidently iniciado.")
    print(f"Intervalo configurado: {INTERVALO_SEGUNDOS} segundos ({INTERVALO_SEGUNDOS / 3600:.1f} horas)")

    # Pequeña espera inicial para dar tiempo a que 'evidently' termine de levantar
    time.sleep(30)

    while True:
        generar_y_subir_reporte()
        print(f"Esperando {INTERVALO_SEGUNDOS} segundos para el próximo ciclo...")
        time.sleep(INTERVALO_SEGUNDOS)