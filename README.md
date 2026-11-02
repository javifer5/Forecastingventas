# Forecasting de ventas

Proyecto de Machine Learning para explorar, preparar, entrenar y servir modelos de pronostico de ventas.

## Estructura

- `data/raw/`: datos originales, sin modificar.
- `data/processed/`: datos limpios y transformados.
- `data/external/`: fuentes externas.
- `notebooks/`: analisis exploratorio, preprocesamiento, entrenamiento y evaluacion.
- `src/`: codigo reutilizable del proyecto.
- `models/`: modelos serializados y artefactos generados.
- `app/`: aplicacion de Streamlit.
- `reports/figures/`: graficas exportadas.
- `docs/`: documentacion tecnica.
- `tests/`: pruebas automatizadas.

## Instalacion

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Flujo recomendado

1. Colocar los datos fuente en `data/raw/`.
2. Ejecutar los notebooks en orden numerico.
3. Guardar el modelo entrenado en `models/`.
4. Iniciar la app:

```bash
streamlit run app/streamlit_app.py
```
