# Documentacion

## Convenciones de datos

- Las fechas deben almacenarse en una columna `date` o adaptarse en la etapa de carga.
- La variable objetivo recomendada es `sales`.
- Los datos originales no se modifican; toda transformacion se guarda en `data/processed/`.

## Reproducibilidad

- Fijar semillas aleatorias en los experimentos.
- Registrar metricas y parametros en los notebooks.
- Versionar codigo y configuracion, no datos sensibles ni modelos pesados.
