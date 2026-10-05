# Customer Intelligence Suite — Retail Banking

## Verificación del modelo de riesgo

- Instalación desde `requirements.txt` en un entorno nuevo del proyecto: correcta.
- Comprobación de dependencias (`pip check`): correcta.
- 9 pruebas automatizadas: correctas.
- Entrenamiento completo desde el Parquet existente: correcto.
- Métricas reproducidas entre entorno de desarrollo y entorno nuevo dentro de tolerancia numérica de 1e-12.
- Recarga del modelo: predicciones idénticas.
- Puntuación de 682 préstamos desde Parquet: correcta; scores finitos en [0,1].
- Puntuación desde CSV sin etiquetas: idéntica a Parquet.
- Predicciones del bloque reservado coinciden con el modelo guardado.

No se reconstruyó PostgreSQL ni se reejecutaron los notebooks 01/02.
El notebook 04 es una interfaz al mismo módulo; se revisó su sintaxis, sin ejecutarlo en Jupyter.
