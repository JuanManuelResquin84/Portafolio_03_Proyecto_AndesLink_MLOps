import pandas as pd
import os
from evidently import Report
from evidently.presets import DataDriftPreset

path_csv = "/app/data/logs.csv"
path_html = "/app/data/index.html"

if os.path.exists(path_csv):
    df = pd.read_csv(path_csv)
    if len(df) > 5:
        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=df.iloc[:5], current_data=df)
        result.save_html(path_html)
        print("Reporte generado exitosamente en index.html")
    else:
        print("Se necesitan mas registros en logs.csv")
else:
    print("No se encontro el archivo:", path_csv)