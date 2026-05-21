# **PROYECTO DE PREDICCIÓN DE CHURN - AndesLink**

### Carrera: **Ciencia de Datos e IA - 2do Año**
### Alumno: **Juan Manuel Resquin**
### Materia: **Laboratorio de Minería de Datos**

* [Repositorio del Proyecto en DagsHub](https://dagshub.com/JuanManuelResquin84/Proyecto_AndesLink)
* [Registro de Experimentos (MLflow)](https://dagshub.com/JuanManuelResquin84/Proyecto_AndesLink.mlflow)

# **OBJETIVO**
### Este proyecto implementa un sistema de MLOps de extremo a extremo para predecir la pérdida de clientes (Churn) en AndesLink. El sistema no solo analiza datos, sino que gestiona el ciclo de vida del modelo mediante el registro de experimentos, control de versiones y despliegue de una API.
### **Limitaciones del Proyecto:** El dataset utilizado es sintético, por lo que carece de factores externos reales (estacionalidad, cambios macroeconómicos o acciones de la competencia) que influyen en el Churn. El modelo es una base sólida, pero requeriría re-entrenamiento con datos reales de producción para ajustar los umbrales de decisión.

# **TECNOLOGÍAS Y ENTORNO**

* **Entorno:** 
    * churn_env2 (utilizado para el Notebook de 01_AndesLink.ipynb), Optimizado para MLflow y producción.

* **Seguimiento de Experimentos:** MLflow integrado con DagsHub.

* **Modelado:** Scikit-Learn (Gradient Boosting Classifier, CatBooosting Cassifiera, Ada Boost Classifier).

* **API Framework:** FastAPI con documentación interactiva (Swagger).


# **ARQUITECTURA DE SOLUCIÓN**

```mermaid
¡Totalmente! Qué buen ojo tuviste. Si dejas ese diagrama de Mermaid como estaba antes, el profesor se va a confundir porque ahí todavía figura el notebook de EDA, dice que entrenás con PyCaret (cuando ahora usás scikit-learn puro en tus scripts) y el nombre del modelo quedó genérico como model.pkl.

Hay que actualizar el diagrama para que refleje tu arquitectura de producción real: la separación modular en los tres scripts dentro de la carpeta src (prepare.py, train.py, main.py) y la inclusión del escalador que le sumaste al pipeline.

Aquí tenés el código de Mermaid corregido y adaptado a tu arquitectura definitiva:

Fragmento de código
%%{init: {'theme': 'dark'}}%%
graph TD
    DATA_RAW[(churn_sintetico.csv)] --> P_PREP["src/prepare.py (Procesamiento)"]
    P_PREP --> DATA_PROC[(churn_procesado.csv)]
    
    DATA_PROC --> P_TRAIN["src/train.py (Entrenamiento Scikit-Learn)"]
    
    subgraph MLOps_Stack ["Gestión y Registro en DagsHub"]
        TRACK["MLflow Tracking (Experimentos)"]
        REG["MLflow Model Registry (V1 Definitivo)"]
    end
    
    P_TRAIN --> TRACK
    TRACK --> REG
    
    subgraph Artefactos ["Artefactos Versionados en DVC"]
        M1["modelo_churn_GBC_andeslink.pkl"]
        S1["scaler_andeslink.pkl"]
    end
    
    REG --> M1
    P_TRAIN --> S1

    M1 --> API["src/main.py (FastAPI App)"]
    S1 --> API
    API --> RES{"Predicciones en /docs"}
```

# **FASE 1:** Análisis Exploratorio (EDA) (01_AndesLink.ipynb)

### Se realizó una auditoría de calidad sobre 5000 registros.


## **Distribución de Clientes**

### Este gráfico mide el desequilibrio de clases en el dataset.

### Donde la gran mayoría de los clientes se mantienen (0), pero tenemos una cantidad significativa de abandonos (1).

### **Interpretación:** Visualmente, parece que el churn representa aproximadamente un 30-35% de la base total.

### Para entrenar un modelo de Machine Learning más adelante, este "desbalance" es importante; si el grupo 1 fuera mucho más pequeño, tendrías que usar técnicas especiales (como sobremuestreo), para que el modelo aprenda bien a identificar a los que se van.

![distribucion_clientes](reports/distribucion_clientes.png)


# **Promedio de Tickets de Soporte vs. Churn**

### Este gráfico muestra la relación entre la frustración técnica/administrativa y la salida del cliente.

### Existe una correlación clara. Los clientes que abandonan generan, en promedio, más tickets de soporte (cerca de 1.9) que los que se quedan (aprox. 1.6).

### La fricción en el servicio es un disparador del abandono.

### Las pequeñas líneas negras arriba de las barras (barras de error) son cortas, lo que significa que esta diferencia es estadísticamente significativa y no es resultado del azar.

![tickets_soporte_vs_churn](reports/tickets_soporte_vs_churn.png)

# **Meses de Antigüedad vs. Churn**

### Aquí comparamos la "lealtad" en tiempo. El boxplot muestra la mediana (la línea negra central) y la dispersión del 50% de los datos (la caja azul).

### Osea los clientes que se quedan (0) tienen una mediana de antigüedad notablemente más alta (cerca de los 40 meses), que los que se van (1) (cuya mediana está cerca de los 30 meses).

### La hipótesis de que "los nuevos se van más" parece ser correcta. 
### La caja del grupo 1 está más abajo en el eje Y, lo que indica que el riesgo de abandono es mayor durante los primeros dos años y medio de contrato.

![boxplot_antiguedad_churn](reports/boxplot_antiguedad_churn.png)

# **Estrategia:**

### Tenemos un problema de retención concentrado en los clientes más nuevos (menos de 30 meses), y aquellos que experimentan problemas técnicos recurrentes (reflejado en el mayor volumen de tickets). 

### Para bajar el churn, deberíamos enfocarnos en mejorar la experiencia de soporte.

### Y crear planes de fidelización para los clientes que llevan menos de dos años con nosotros. 


# **FASE 2:** 
# **Conclusión Final**
### Para ir por el camino más conservador (priorizando el cuidado del presupuesto y evitando regalar promos a gente que no se iba), tome la mejor opción que es el **modelo Gradient_Boosting_Train_V1**. Con el ajuste de hiperparámetros y el umbral óptimo de 0.45, logrando un balance ideal para una estrategia de **"costo controlado"**.

### **Justificación detallada de por qué este es el modelo para el negocio:**
> Es el más "Tacaño" y eficiente (Mayor Precisión). Evitando **"gastar tanto en campaña"**. La métrica clave que controla el desperdicio de dinero es la Precisión.

> La versión V1 del modelo GBC alcanzó una **Precisión de 0.5714 (57.14%)**.

> **Significa en la vida real:** De cada 100 personas a las que el modelo les ponga el cartel de "¡Ojo, este cliente se va!", más de 57 realmente se van a ir.

> **El beneficio económico:** Minimizás los Falsos Positivos. No desperdiciás presupuesto reglando descuentos o llamando a clientes fieles que no tenían ninguna intención de abandonar el servicio.

> Se mejoró drásticamente la cobertura (Adiós a la Trampa del Recall). Con la **versión V1**, el **Recall subió a 0.5529 (55.29%)**.

> **Logrando romper el cuello de botella:** ahora atrapamos a más de la mitad de los clientes en riesgo real de fuga, pero sin volvernos masivos ni permisivos. Mantenemos el tiro de precisión.

> **Consistencia Estadística (Kappa Alto y Buen Accuracy)**

> **El Accuracy (0.7070)** demuestra que el modelo acierta globalmente en 7 de cada 10 casos.

> **Lo más importante:** el índice Kappa llegó a 0.3419. Esto demuestra que las predicciones del modelo son sólidas y bien fundamentadas por los patrones de los datos, y no fruto del azar o del desbalance de clases.

# **Resumen del plan estratégico con GBC V1:**

> **Campañas Quirúrgicas:** Contactarás a un volumen moderado de clientes **(Recall ~55%)**, asegurando que el equipo de retención de AndesLink no se sature.

> **Alta Efectividad:** La mayoría de los contactos serán sobre riesgos reales **(Precisión ~57%)**.

> **Optimización del ROI:** El dinero que ahorramos al no emitir falsas alarmas, nos permite financiar promociones de mayor impacto o valor para los clientes que sí están en la cuerda floja, maximizando la tasa de éxito de la campaña.

# **Conclusión:** Si la prioridad número uno de la empresa es optimizar cada peso invertido en retención, el **modelo Gradient_Boosting_Train_V1** es la elección financiera y técnica más inteligente.

![tabla_comparativa_modelos](reports/tabla_comparativa_modelos_final.png)
![tabla_comparativa_modelos](reports/tabla_comparativa_modelos_final2.png)

# **FASE 3:** Implementación de API (FastAPI)

### Desarrolle una API para consultas en tiempo real.
* **Endpoints:** /predict para inferencia inmediata.
* **Normalización:** Los datos de entrada pasan por el mismo pipeline de escalado registrado en el entrenamiento.
* **Probabilidad de Fuga:** ~90.47% (Ejemplo validado).
* **Acción:** Generación automática de alerta para retención.

### **ARTEFACTOS REGISTRADOS**
### Los componentes críticos se encuentran versionados:
* **model.pkl:** El motor de decisión (GradientBoostingClassifier).

![1](reports/1.png)
![2](reports/2.png)
![3](reports/3.png)
![4](reports/4.png)

# **ANEXO: CÓMO REPRODUCIR EL PROYECTO**

1. **Clonar el repositorio:** 
* `git clone https://github.com/JuanManuelResquin84/Portafolio_03_Proyecto_AndesLink_MLOps.git`
(Asegurate de estar en la carpeta del proyecto) 
* `cd Portafolio_03_Proyecto_AndesLink_MLOps`

2. **Configurar el entorno:** 
* `conda create -n churn_env2 python=3.10 -y`
* `conda activate churn_env2`
* `pip install -r requirements_churn_env2.txt`

3. **Descargar Datos y Modelos (Sincronización):**
* `dagshub download JuanManuelResquin84/Proyecto_AndesLink . .`

4. **Ejecutar entrenamiento:** 
Ejecutar el Pipeline de Entrenamiento (Opcional):
Si desea volver a procesar los datos y reentrenar el modelo generando un nuevo experimento en MLflow, ejecute en orden:
`python src/prepare.py`
`python src/train.py`

5. **Lanzar la API de Inferencia:** 
* `python src/main.py`

6. **Probar API: ingresando a la Interfaz de Prueba** 
* `http://localhost:8000/docs`
(Una vez que el servidor esté activo, acceda a la documentación interactiva para realizar predicciones en tiempo real enviando un JSON de prueba)