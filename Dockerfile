# Usamos una imagen liviana oficial de Python
FROM python:3.10-slim

# Seteamos el directorio de trabajo adentro del contenedor
WORKDIR /app

# Instalamos dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiamos el archivo de requirements al contenedor
COPY requirements_churn_env2.txt .

# Instalamos las librerías de Python
RUN pip install --no-cache-dir -r requirements_churn_env2.txt

# Copiamos las carpetas de código, datos y modelos al contenedor
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Exponemos el puerto 8000 (el que usa FastAPI)
EXPOSE 8000

# Comando para arrancar la API al encender el contenedor
CMD ["python", "src/main.py"]