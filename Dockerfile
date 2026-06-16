FROM python:3.10-slim
WORKDIR /app
COPY requirements_churn_env2.txt . 
RUN pip install --no-cache-dir -r requirements_churn_env2.txt
COPY . .