# Customer Intelligence Suite — Retail Banking

**Etapa actual:** análisis de datos y modelo base de riesgo crediticio.  
**Próximos pasos:** evaluar la estabilidad del modelo y completar la segmentación de clientes.

Prototipo educativo con datos históricos Berka: transforma datos bancarios, construye datasets analíticos y entrena un modelo base de riesgo para préstamos finalizados.

**Estado: versión preliminar.** Incluye entrenamiento reproducible, evaluación temporal retrospectiva, modelo guardado y puntuación por lote. La segmentación está iniciada, pero todavía no terminada. No es un motor de aprobación de créditos ni una probabilidad de impago calibrada.

## Inicio rápido

Usar Python 3.12 desde la raíz del proyecto:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.credit_risk_model train
```

El entrenamiento usa `data/processed/credit_risk.parquet`, ya generado por el notebook 02. No requiere PostgreSQL ni credenciales. Genera:

- `reports/credit_risk_report.md`: resultados y límites de interpretación.
- `reports/credit_risk_metrics.json`: métricas, fechas, versiones y huella del archivo de entrada.
- `reports/holdout_predictions.csv`: predicciones del bloque de evaluación.
- `reports/model_coefficients.csv`: coeficientes estandarizados.
- `models/credit_risk_baseline.joblib`: pipeline ajustado solo con entrenamiento.

## Puntuar un lote

```sh
python -m src.credit_risk_model score --input data/processed/credit_risk.parquet --output reports/batch_scores.csv
```

Acepta Parquet o CSV con `loan_id` único y las diez variables de la tabla siguiente. No necesita etiquetas. El score es experimental y el umbral fijo de 0,5 es ilustrativo. Puntuar el dataset completo incluye observaciones usadas al entrenar; ese archivo **no es una evaluación**. Para medir desempeño, usar exclusivamente el informe del bloque reservado. Cargar únicamente modelos propios o de confianza.

| Variable | Significado |
|---|---|
| `loan_amount` | Importe contractual |
| `loan_duration` | Duración contractual en meses |
| `account_tenure_days` | Antigüedad de cuenta al originarse el préstamo |
| `preloan_avg_balance` | Saldo medio observado antes del préstamo |
| `preloan_min_balance` | Saldo mínimo previo |
| `preloan_balance_std` | Desviación de saldos previos |
| `preloan_negative_balance_ratio` | Proporción de transacciones con saldo negativo |
| `preloan_transactions_per_month` | Actividad mensual en el intervalo observado |
| `preloan_monthly_inflow` | Entradas mensuales en el intervalo observado |
| `preloan_monthly_outflow` | Salidas mensuales en el intervalo observado |

Los importes conservan la unidad monetaria del dataset histórico; no se convierten a moneda actual.
En nuevos lotes, construir las variables con las mismas definiciones del notebook 02 y con transacciones estrictamente anteriores a la originación. La interfaz de puntuación verifica el esquema, no reconstruye el historial fuente.

## Diseño de evaluación

Se conservan para modelar solo contratos A/B: A=0 y B=1. C/D siguen sin objetivo definitivo. Se ordenan por originación y se reserva aproximadamente el último 25%, manteniendo juntas fechas iguales. El modelo es una regresión logística regularizada con imputación por mediana y escalado ajustados exclusivamente en entrenamiento. No hay búsqueda de hiperparámetros ni selección del umbral sobre el bloque reservado. Se compara contra la frecuencia de impago de entrenamiento.

Las fechas, identificadores, estado, etiqueta y variables demográficas no son predictores. El modelo usa condiciones contractuales conocidas a la originación, por lo que no representa una evaluación anterior a definir la oferta.

El conjunto tiene 682 préstamos, pero solo 234 finalizados, con 31 impagos. El bloque reservado contiene únicamente 4 impagos. La selección de contratos cerrados introduce sesgo de maduración: los resultados son exploratorios. Tampoco se conocen las fechas exactas en que cada desenlace pasó a estar disponible, por lo que el corte por originación no constituye una simulación histórica de despliegue.

## Reconstrucción desde los datos fuente

La ejecución rápida anterior parte del dataset procesado. Para reconstruirlo:

1. Crear una base PostgreSQL vacía dedicada al proyecto.
2. Copiar `.env.example` a `.env` y configurar `DATABASE_URL` con el driver `postgresql+psycopg`.
3. Colocar los ocho archivos Berka `.asc` en `data/raw/`.
4. Seguir los comandos de creación de esquema y carga de [docs/README.md](docs/README.md). **Ese procedimiento borra y recrea las tablas del proyecto: usar solo una base dedicada.**
5. Instalar Jupyter si se desea trabajar en notebooks: `python -m pip install jupyterlab`.
6. Ejecutar, desde `notebooks/`, `01_eda.ipynb` y `02_analytical_dataset.ipynb`, en orden y con el entorno anterior. El segundo exporta los Parquet.
7. Volver a la raíz y ejecutar el entrenamiento.

La nueva etapa de modelamiento se verificó con el Parquet existente. No se reconstruyó la base ni se volvieron a ejecutar los notebooks 01 y 02 durante este cierre.

## Estructura y continuación

- `src/etl.py`, `database/schema.sql`: preparación y carga.
- `notebooks/01_eda.ipynb`: análisis exploratorio.
- `notebooks/02_analytical_dataset.ipynb`: datasets de clientes y riesgo.
- `notebooks/03_customer_segmentation.ipynb`: trabajo de segmentación pendiente.
- `notebooks/04_credit_risk_modeling.ipynb`: entrenamiento y evaluación del modelo de riesgo.
- `src/credit_risk_model.py`: entrenamiento y puntuación.
- `tests/test_credit_risk_model.py`: controles de objetivos, fechas y separación.
- [docs/NEXT_SESSION.md](docs/NEXT_SESSION.md): estado guardado y próximos pasos.

Origen: Berka / PKDD'99; documentación de la fuente en `docs/`. Este proyecto no añade una licencia de redistribución a los datos originales. El repositorio presenta el estado actual del proyecto; los próximos pasos están documentados en `docs/NEXT_SESSION.md`.
