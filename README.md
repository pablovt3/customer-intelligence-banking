# Customer Intelligence Suite — Retail Banking

**Etapa actual:** análisis de datos y modelo base de riesgo crediticio.  
**Próximos pasos:** evaluar la estabilidad del modelo y completar la segmentación de clientes.

Prototipo educativo con datos históricos Berka: transforma datos bancarios, construye datasets analíticos y entrena un modelo base de riesgo para préstamos finalizados.

**Estado: versión preliminar.** Incluye entrenamiento reproducible, evaluación temporal retrospectiva, modelo guardado y puntuación por lote. La segmentación está iniciada, pero todavía no terminada. No es un motor de aprobación de créditos ni una probabilidad de impago calibrada.

## Flujo del proyecto

```text
Archivos Berka (.asc)
        ↓ extracción, limpieza y transformación — src/etl.py
PostgreSQL (8 tablas relacionadas)
        ↓ análisis exploratorio — notebook 01
Datasets analíticos de clientes y préstamos — notebook 02
        ↓ exportación a Parquet
Segmentación de clientes (en desarrollo) / Modelo base de riesgo crediticio
        ↓
Evaluación, informes y puntuación por lote
```

La base relacional y el proceso ETL son la primera etapa del proyecto. El modelamiento consume los datasets que se construyen sobre esa base. Hay dos formas de ejecutar el trabajo:

- **Recorrido completo:** configurar PostgreSQL, crear el esquema, cargar los archivos fuente y ejecutar los notebooks. Ver [Configuración de PostgreSQL y ejecución del ETL](#configuración-de-postgresql-y-ejecución-del-etl).
- **Modelo con datos ya preparados:** usar los Parquet incluidos para entrenar y puntuar sin conectarse a PostgreSQL, como se explica a continuación.

## Inicio rápido — modelo con datos preparados

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

## Configuración de PostgreSQL y ejecución del ETL

Los comandos siguientes se ejecutan desde la raíz del repositorio, salvo cuando se indica cambiar a `notebooks/`. Requieren Python 3.12, las dependencias de `requirements.txt`, un servidor PostgreSQL en ejecución y las herramientas cliente `psql`, `createuser` y `createdb` disponibles. El proyecto no instala ni inicia PostgreSQL automáticamente.

### 1. Preparar el entorno de Python

Si todavía no ejecutaste el inicio rápido:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Crear un usuario y una base dedicada

Ejemplo para PostgreSQL local en el puerto 5432. Reemplazar `postgres` por el administrador de la instalación si tiene otro nombre. Los comandos solicitan las contraseñas de forma interactiva; no se escriben en el historial de la terminal.

```sh
createuser --host=localhost --port=5432 --username=postgres --pwprompt credit_app
createdb --host=localhost --port=5432 --username=postgres --owner=credit_app customer_intelligence
```

Estos comandos se ejecutan una sola vez. Si ya existen el usuario y la base dedicada, reutilizarlos. `credit_app` será propietario de la base y ejecutará el esquema y las cargas; no necesita ser superusuario.

### 3. Configurar la conexión

Crear la configuración local sin sobrescribir un `.env` existente:

```sh
python - <<'PYENV'
from pathlib import Path
example = Path('.env.example')
local = Path('.env')
if local.exists():
    print('.env ya existe; revisa su configuración local.')
else:
    local.write_text(example.read_text())
    print('.env creado a partir de la plantilla.')
PYENV
```

Editar `.env` con el usuario, contraseña, host, puerto y nombre de la base. El siguiente valor es exclusivamente una plantilla; `REEMPLAZAR_CLAVE` no es una credencial real:

```dotenv
DATABASE_URL=postgresql+psycopg://credit_app:REEMPLAZAR_CLAVE@localhost:5432/customer_intelligence
```

Si la contraseña incluye caracteres reservados de una URL, codificarlos mediante percent-encoding antes de incorporarla. `src/database.py` lee el `.env` de la raíz mediante `python-dotenv` y crea el motor SQLAlchemy con el driver `psycopg`. Una variable `DATABASE_URL` ya exportada en el entorno tiene prioridad sobre el archivo. `.env` y sus variantes están excluidos de Git; solo se publica `.env.example`.

Comprobar la conexión sin imprimir credenciales:

```sh
python - <<'PYCONNECT'
from sqlalchemy import text
from src.database import engine
with engine.connect() as connection:
    assert connection.execute(text('SELECT 1')).scalar_one() == 1
print('Conexión a PostgreSQL correcta.')
PYCONNECT
```

### 4. Crear el esquema relacional

[database/schema.sql](database/schema.sql) define claves primarias, claves foráneas y restricciones de valores y consistencia.

**Este archivo elimina y recrea las ocho tablas, utilizando `DROP TABLE ... CASCADE`. Ejecutarlo únicamente sobre la base dedicada al proyecto; elimina cualquier carga anterior y puede eliminar objetos dependientes.**

```sh
python - <<'PYSCHEMA'
from pathlib import Path
from src.database import engine
with engine.begin() as connection:
    connection.exec_driver_sql(Path('database/schema.sql').read_text())
print('Esquema creado.')
PYSCHEMA
```

### 5. Ejecutar la extracción, transformación y carga

El ETL lee los archivos separados por `;` de `data/raw/`. El mapeo de archivos a tablas y los volúmenes del extracto incluido son:

| Archivo fuente | Tabla PostgreSQL | Registros |
|---|---|---:|
| `district.asc` | `district` | 77 |
| `account.asc` | `account` | 4.500 |
| `client.asc` | `client` | 5.369 |
| `disp.asc` | `disp` | 5.369 |
| `card.asc` | `card` | 892 |
| `order.asc` | `standing_order` | 6.471 |
| `loan.asc` | `loan` | 682 |
| `trans.asc` | `bank_transaction` | 1.056.320 |

Las cuentas y clientes se relacionan con distritos; `disp` vincula clientes y cuentas e identifica al titular o autorizado. Las tarjetas se vinculan a `disp`, mientras que préstamos, órdenes permanentes y transacciones se vinculan a cuentas.

[src/etl.py](src/etl.py) realiza estas operaciones:

- **Extracción:** lectura de los ocho archivos; interpreta `?` como dato faltante en distritos.
- **Limpieza:** elimina espacios en los campos de texto tratados y convierte cadenas vacías en valores faltantes.
- **Transformación:** convierte fechas y normaliza categorías de frecuencia de cuenta, órdenes y transacciones. Las categorías no contempladas provocan un error; los estados de préstamo conservan los códigos A/B/C/D.
- **Demografía:** decodifica fecha de nacimiento y sexo registrado en `birth_number`, conservando el código original. Calcula edad al último día observado en las transacciones, no a la fecha actual. La interpretación del siglo 1900 es una hipótesis explícita del extracto histórico.
- **Carga:** inserta primero las entidades referenciadas y después las dependientes, respetando claves foráneas. Las transacciones se cargan en bloques de 10.000 registros.
- **Validación:** compara el número de filas de cada tabla con el archivo de entrada transformado; PostgreSQL aplica además las restricciones del esquema. Esto no sustituye los controles analíticos del EDA.

Ejecutar sobre las tablas vacías creadas en el paso anterior:

```sh
python -m src.etl
```

Al finalizar, el proceso muestra los registros cargados y validados por tabla y el mensaje `ETL pipeline completed successfully.`

**La carga usa inserciones acumulativas (`append`): no es incremental ni idempotente.** No ejecutarla otra vez sobre tablas pobladas, porque chocará con las claves existentes. Si la carga falla a mitad, algunas tablas pueden haber quedado cargadas: la ejecución completa no está envuelta en una única transacción. Para reconstruir, corregir la causa, recrear el esquema en la base dedicada y repetir el ETL.

### 6. Ejecutar el análisis y exportar los datasets

```sh
python -m pip install jupyterlab
cd notebooks
python -m jupyterlab
```

Elegir el kernel del entorno del proyecto y ejecutar las celdas en orden:

1. `01_eda.ipynb`: estructura, calidad, integridad, actividad bancaria y hallazgos exploratorios.
2. `02_analytical_dataset.ipynb`: una fila por cuenta en `customer_360` y una por préstamo en `credit_risk`; controles temporales y definición del objetivo. Exporta ambos datasets a `data/processed/` como Parquet.
3. `04_credit_risk_modeling.ipynb`: entrenamiento y evaluación del modelo base. El notebook 03 de segmentación sigue en desarrollo y no es un requisito para ejecutar el modelo de riesgo.

Alternativamente, después de exportar los datasets, volver a la raíz y usar el módulo de entrenamiento:

```sh
cd ..
python -m src.credit_risk_model train
```

### Problemas frecuentes

| Síntoma | Qué revisar |
|---|---|
| Conexión rechazada | Servidor iniciado, host y puerto de `DATABASE_URL`. |
| Fallo de autenticación | Usuario, contraseña, codificación de caracteres y reglas de autenticación de PostgreSQL. |
| Base inexistente | Nombre de la base y ejecución de `createdb`. |
| Tabla inexistente | Ejecutar el esquema antes del ETL. |
| Clave primaria duplicada | La carga ya se ejecutó o quedó incompleta; reconstruir en la base dedicada. |
| Notebook no encuentra `src` | Abrirlo desde `notebooks/` con el kernel del entorno del proyecto. |

**Alcance de la verificación:** el entrenamiento y la puntuación se comprobaron en un entorno nuevo usando el Parquet existente. Esta documentación se contrastó con el ETL y el esquema actuales; no se borró ni reconstruyó la base local, ni se volvieron a ejecutar los notebooks 01/02. Los detalles de la transformación demográfica se documentan en [docs/README.md](docs/README.md).

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
