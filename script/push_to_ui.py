import pandas as pd
import joblib
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.ui.workspace import RemoteWorkspace

# 1. Cargar datos
ref = pd.read_csv("/app/data/churn_procesado.csv")
logs = pd.read_csv("/app/data/logs.csv")
cols = joblib.load("/app/models/X_columns.pkl")

# 2. Armonizar
logs_transformed = pd.get_dummies(logs).reindex(columns=cols, fill_value=0)

# 3. Crear objetos Dataset (requerido en 0.7.x)
ref_dataset = Dataset.from_pandas(ref, data_definition=DataDefinition())
current_dataset = Dataset.from_pandas(logs_transformed, data_definition=DataDefinition())

# 4. Crear reporte con DataDriftPreset y DataSummaryPreset
report = Report([
    DataSummaryPreset(),
    DataDriftPreset()
])

# 5. Ejecutar el reporte
run = report.run(current_dataset, ref_dataset)

# 6. Conectar a la UI de Evidently (puerto 8001)
ws = RemoteWorkspace("http://evidently:8000")

# 7. Crear proyecto
project = ws.create_project("Monitoreo_AndesLink")

# 8. Agregar el run al proyecto
ws.add_run(project.id, run)

print("Reporte subido a la UI correctamente.")
print(f"Recarga http://localhost:8001 para ver el proyecto.")