# Customer Intelligence Suite — Retail Banking

## Riesgo crediticio — evaluación retrospectiva

**Próximos pasos:** evaluar la estabilidad temporal del modelo y revisar la definición del objetivo antes de ampliar el análisis.

Regresión logística regularizada con 10 variables de contrato e historial previo al préstamo.
La referencia asigna a todos la frecuencia de impago del conjunto de entrenamiento.
El modelo guardado se entrenó exclusivamente con el bloque de entrenamiento; no se reajustó con la evaluación.

## Población y separación

- Total: 682 préstamos; 448 activos excluidos del entrenamiento y evaluación.
- Entrenamiento: 175 préstamos, 27 impagos, 1993-07-05 a 1996-08-22.
- Evaluación: 59 préstamos, **4 impagos**, 1996-09-11 a 1997-12-28.
- Corte por fecha de originación, aproximadamente 75/25; una misma fecha nunca se divide.
- A=0 y B=1. C/D no tienen desenlace definitivo y no se imputan.

## Resultados fuera de entrenamiento

| Métrica | Regresión logística | Referencia |
|---|---:|---:|
| ROC AUC ↑ | 0.8818 | 0.5000 |
| Average precision ↑ | 0.6101 | 0.0678 |
| Brier score ↓ | 0.0472 | 0.0707 |
| Precisión a 0,5 | 1.0000 | 0.0000 |
| Sensibilidad a 0,5 | 0.5000 | 0.0000 |

Matriz [verdaderos negativos, falsos positivos, falsos negativos, verdaderos positivos]:
`[55, 0, 2, 2]`.
Umbral fijo de 0,5, sin optimizar con la evaluación y sin convertirlo en política de aprobación.

## Alcance y límites

Esta entrega completa un flujo educativo: datos analíticos → entrenamiento → evaluación → modelo guardado → puntuación por lote.
Con solo 4 impagos en evaluación, las métricas son inestables; no prueban capacidad de generalización.
La selección de contratos finalizados introduce sesgo de maduración y de duración.
Separar por originación no asegura que los desenlaces de entrenamiento ya fueran conocidos en esa fecha:
no se dispone de fechas exactas de observación del impago. Esto es una evaluación retrospectiva,
no una simulación histórica de despliegue ni una probabilidad de impago a un horizonte fijo.
El score no está calibrado para uso bancario real.

Imputación y escalado se ajustan solo con entrenamiento. Estado del préstamo, objetivo, identificadores,
fechas y variables demográficas quedan fuera de los predictores. Se comprueban fechas de historial,
pero esas comprobaciones no sustituyen una nueva auditoría de las transacciones fuente.
Las variables mensuales heredadas usan el intervalo entre primera y última transacción observada.
Los coeficientes estandarizados describen asociaciones, no efectos causales.

## Artefactos

- `credit_risk_metrics.json`: métricas, población, versiones y huella del dataset.
- `holdout_predictions.csv`: predicciones fuera de entrenamiento y etiquetas reales.
- `model_coefficients.csv`: coeficientes por variable.
- `../models/credit_risk_baseline.joblib`: pipeline entrenado y metadatos.

La segmentación de clientes es la siguiente línea de desarrollo del proyecto y se trabajará por separado del modelo de riesgo.
