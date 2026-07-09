"""
Genera tráfico sintético variado contra /predict para asegurar que logs.csv
cubra todas las categorías vistas en entrenamiento (evita drift artificial
por columnas dummy que nunca aparecen en los logs).
"""
import requests
import random

API_URL = "http://localhost:8000/predict"

# Valores REALES confirmados en churn_sintetico.csv
CONTRACT_TYPES = ["mensual", "anual", "bianual"]
PAYMENT_METHODS = ["credito", "debito", "efectivo", "transferencia"]
INTERNET_SERVICES = ["cable", "fibra", "movil", "ninguno"]
REGIONS = ["centro", "norte", "oeste", "sur"]

N_REQUESTS = 1000


def cliente_aleatorio():
    return {
        "tenure_months": random.randint(0, 72),
        "monthly_charge": round(random.uniform(20, 300), 2),
        "total_charges": round(random.uniform(0, 20000), 2),
        "support_tickets": random.randint(0, 8),
        "late_payments": random.randint(0, 6),
        "avg_monthly_usage_gb": round(random.uniform(5, 3500), 2),
        "contract_type": random.choice(CONTRACT_TYPES),
        "payment_method": random.choice(PAYMENT_METHODS),
        "internet_service": random.choice(INTERNET_SERVICES),
        "has_streaming": random.randint(0, 1),
        "has_security_pack": random.randint(0, 1),
        "num_products": random.randint(1, 8),
        "region": random.choice(REGIONS),
        "customer_age": random.randint(18, 80),
        "is_promo": random.randint(0, 1),
    }


if __name__ == "__main__":
    exitosos, fallidos = 0, 0
    for i in range(N_REQUESTS):
        payload = cliente_aleatorio()
        try:
            r = requests.post(API_URL, json=payload, timeout=5)
            if r.status_code == 200:
                exitosos += 1
            else:
                fallidos += 1
                print(f"Fallo #{i}: {r.status_code} - {r.text}")
        except Exception as e:
            fallidos += 1
            print(f"Error de conexión #{i}: {e}")

    print(f"\nCompletado: {exitosos} exitosos, {fallidos} fallidos de {N_REQUESTS} requests.")
