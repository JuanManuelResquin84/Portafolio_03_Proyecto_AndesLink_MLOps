# **Materia:** Laboratorio de Minería de Datos
# **Alumno:** Juan Manuel Resquin
# **Proyecto Académico:** 
# MLOps Local para un Modelo de Machine Learning
# **Caso de Uso:** 
# Predicción de abandono de clientes (Customer Churn)

# Empresa Simulada Contratante: **AndesLink Servicios Digitales S.A.** 
### Es una empresa ficticia de suscripción mensual que necesita anticipar el abandono de clientes para activar campañas de retención. La compañía contrata al alumno como proveedor tecnológico para diseñar, desarrollar, desplegar y monitorear una solución completa de Machine Learning bajo prácticas de MLOps en una computadora local.

# Documento de referencia para las tres instancias de evaluación:
## 1) Primer parcial – **Entrenamiento**
## 2) Segundo parcial – **Despliegue**
## 3) Examen final – **Monitoreo y presentación técnica**

# **1. Contexto de negocio**
## AndesLink Servicios Digitales S.A. comercializa planes de suscripción para servicios digitales orientados a consumidores finales. Durante los últimos trimestres, la empresa ha detectado una tasa creciente de cancelación voluntaria de clientes. El directorio entiende que la pérdidade clientes no solo afecta los ingresos recurrentes, sino también el costo de adquisición de nuevos usuarios, la estabilidad del flujo de caja y la eficiencia de sus campañas comerciales.
## Con el objetivo de tomar decisiones más oportunas, la empresa desea contar con un modelo de Machine Learning capaz de estimar la probabilidad de churn a partir de variables de comportamiento, antigüedad, facturación y relación con el servicio. Además, exige que la solución no quede limitada al entrenamiento del modelo: debe poder ejecutarse localmente, exponerse mediante una API, consumirse desde una interfaz gráfica simple y contar con monitoreo técnico y de datos. La empresa contrata al alumno como responsable de construir una solución end to end en un entorno local,  con trazabilidad, reproducibilidad y organización de proyecto semejantes a las de un escenario profesional.

# **2. Objetivo general del proyecto**
## Diseñar, desarrollar y documentar desde cero una solución local de MLOps para predecir churn de clientes, cubriendo el ciclo completo de entrenamiento, despliegue productivo y monitoreo operativo del modelo.

# **3. Objetivos específicos**
* Comprender el problema de negocio y traducirlo a un objetivo analítico medible.
* Preparar un dataset tabular apto para modelado, con justificación de variables y tratamiento de calidad de datos.
* Entrenar y evaluar al menos un modelo supervisado binario, generando un artefacto serializado listo para inferencia.
* Versionar código, datos y experimentos con herramientas acordes a un flujo MLOps local.
* Desplegar el modelo como API y construir una GUI capaz de invocar dicha API.
* Instrumentar monitoreo técnico y monitoreo de drift/datos para una operación observable.
* Comunicar la solución con documentación técnica clara y una exposición final ejecutiva-técnica.

# **4. Caso de uso y alcance funcional**
## El alumno deberá resolver un caso de clasificación binaria orientado a predecir si un cliente abandonará o no el  servicio. La cátedra podrá proveer un dataset de churn. En ausencia de un dataset provisto, podrá utilizarse un  dataset público equivalente, como IBM Telco Customer Churn, o un dataset sintético construido por el alumno,  siempre que la decisión sea debidamente justificada, documentada y mantenga un nivel de complejidad  razonable para la materia. 
## La solución deberá ser desarrollada como si fuese un proyecto profesional real. No se aceptará una entrega limitada a notebooks aislados sin estructura de proyecto, ni una solución que dependa exclusivamente de ejecución manual sin instrucciones reproducibles.
## Resultado esperado de alto nivel
## Al finalizar el proyecto, el alumno debe poder demostrar: un modelo entrenado y serializado, una API de inferencia operativa, una GUI funcional que consuma la API, contenedorización y orquestación local, métricas observables en tablero y un reporte de monitoreo de datos/modelo.

# **5. Stack tecnológico mínimo requerido**
## La solución deberá implementarse en entorno local, preferentemente sobre Windows con Anaconda y Visual  Studio Code. Se espera el uso de las siguientes herramientas, o equivalentes:

# **6. Requisitos generales de arquitectura y organización**
* El proyecto debe estar estructurado en carpetas claras, separando datos, código fuente, modelos, 
pruebas, scripts, documentación y configuración.
* Debe existir un README principal con instrucciones de instalación, ejecución y validación.
* La solución no debe depender de rutas absolutas ni configuraciones frágiles no documentadas.
* El modelo entrenado debe poder cargarse para inferencia sin necesidad de reentrenar en cada ejecución.
* Debe incluirse un archivo de entorno reproducible, por ejemplo environment.yml o equivalente.
* Toda decisión relevante de modelado, despliegue o monitoreo debe estar justificada técnicamente.
* La cátedra valorará especialmente la claridad del flujo end to end, la robustez y la trazabilidad.

# 7. Estructura mínima esperada del repositorio
## La organización exacta podrá variar, pero se espera una estructura equivalente a la siguiente:
* proyecto/
* data/
* models/
* notebooks/
* src/
* tests/
* scripts/
* reports/
* docker-compose.yml
* Dockerfile
* environment.yml
* README.md

# **8. Entrega 1 – Primer parcial: entrenamiento**
## En esta instancia el alumno deberá llegar, como mínimo, hasta el modelo serializado listo para ser utilizado por una futura API. Esta etapa concentra el trabajo de entendimiento del problema, tratamiento de datos, entrenamiento, evaluación y reproducibilidad inicial.

# **8.1 Requerimientos funcionales obligatorios**
* Definir el problema de negocio y la variable objetivo en términos claros.
* Describir el dataset seleccionado, su procedencia, su tamaño, sus columnas relevantes y sus principales limitaciones.
* Realizar análisis exploratorio mínimo con evidencia gráfica o tabular suficiente.
* Aplicar limpieza de datos, tratamiento de faltantes, codificación de variables y transformaciones justificadas.
* Separar conjuntos de entrenamiento y validación/prueba de forma correcta.
* Entrenar al menos un modelo supervisado de clasificación binaria. Se valorará la comparación entre dos o más alternativas.
* Evaluar el modelo mediante métricas coherentes con el problema, por ejemplo accuracy, precision, recall, F1, ROC-AUC o matriz de confusión, justificando la elección.
* Serializar el artefacto final del modelo y cualquier componente de preprocesamiento necesario para inferencia.
* Registrar experimentos o evidencias equivalentes con MLflow y mantener reproducibilidad del flujo con DVC o una estrategia equivalente debidamente justificada.
* Versionar el proyecto con Git y documentar instrucciones de ejecución.

# **8.2 Entregables obligatorios**
* Repositorio del proyecto o paquete entregable con estructura ordenada.
* Informe técnico corto del parcial con problema, dataset, decisiones de preparación, métricas, conclusiones y limitaciones.
* Notebook o evidencia de exploración inicial.
* Scripts o módulos de entrenamiento reutilizables.
* Archivo serializado del modelo final y, si corresponde, del pipeline de preprocesamiento.
* Archivo de entorno reproducible y README de instalación/ejecución

# **8.3 Criterios de aceptación mínimos**
* El modelo debe poder cargarse sin errores desde un script independiente.
* La métrica elegida debe estar claramente explicada y ser coherente con el caso de churn.
* No se aceptará una entrega cuya ejecución dependa únicamente de celdas manuales no documentadas.
* El evaluador debe poder reproducir el entrenamiento siguiendo instrucciones razonables.


