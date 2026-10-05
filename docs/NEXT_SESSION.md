# Customer Intelligence Suite — Retail Banking

## Próximos pasos

## Punto de partida

Se incorporó un flujo completo de entrenamiento, evaluación, persistencia y puntuación por lote, basado en el dataset procesado de la carpeta original del proyecto. No se modificaron los notebooks anteriores, el ETL ni las fuentes sincronizadas del chat.

La definición del objetivo de riesgo crediticio es `target_default`: A=0, B=1, C/D sin etiqueta. Las copias del chat contienen otra definición (`observed_bad_status`); no deben intercambiarse como entradas del modelo. El código rechaza una entrada incompatible.

## Para retomar

1. Leer `reports/credit_risk_report.md` y discutir si el objetivo final seguirá siendo desempeño de contratos cerrados o un evento a horizonte fijo. No reinterpretar el score como PD anual.
2. Revisar estabilidad con validaciones temporales dentro del bloque de entrenamiento. Mantener el bloque reservado sin usar para ajustar parámetros.
3. Revisar la exposición usada para tasas mensuales: actualmente primera a última transacción observada; comparar con apertura de cuenta a originación mediante validación de entrenamiento.
4. Completar perfiles y validación de segmentos en el notebook 03 si se decide ampliar el alcance de Customer Intelligence.
5. Reconstruir el flujo desde una base vacía de prueba y ejecutar notebooks 01/02 para verificar reproducibilidad integral.
6. Antes de publicar, seleccionar la versión original como fuente de verdad y revisar el contenido exacto que se subirá y las condiciones de distribución del dataset.

## Comandos

```sh
python -m unittest discover -s tests -v
python -m src.credit_risk_model train
python -m src.credit_risk_model score --input data/processed/credit_risk.parquet
```

El modelo guardado conserva el entrenamiento original de la evaluación. No fue reentrenado con los 234 préstamos completos. No se creó servicio, despliegue ni automatización para mañana; el trabajo queda documentado para retomar en este chat.
