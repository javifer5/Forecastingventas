# 📈 Pronóstico de ventas — Simulador de noviembre 2025

[![Abrir la app](https://img.shields.io/badge/Abrir%20la%20app-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ventas-simuladas.streamlit.app/)

Proyecto de Machine Learning que predice las **unidades vendidas por día y por producto en noviembre de 2025** y una app web donde se puede **simular qué pasaría si cambia el descuento o el precio de la competencia**.

👉 **Probala acá:** https://ventas-simuladas.streamlit.app/

> Si la app estuvo un tiempo sin visitas, Streamlit la "duerme". Hacé clic en *Yes, get this app back up* y esperá unos segundos.

---

## 🎯 Objetivo

Anticipar la demanda diaria de cada producto en un mes clave (con Black Friday y Cyber Monday) y medir cómo reaccionan las ventas y los ingresos ante cambios de precio y de competencia, para apoyar decisiones de descuentos y de stock.

## ✨ Qué permite hacer la app

- Elegir un **producto**.
- Ajustar el **descuento** de −50 % a +50 %.
- Elegir un **escenario de competencia**: precios actuales, −5 % o +5 % (Amazon, Decathlon y Deporvillage).
- Ver las **unidades e ingresos proyectados**, el precio promedio y el descuento promedio.
- Ver la **predicción diaria** en un gráfico, con el dato de Black Friday (28/11), y el detalle día por día en una tabla.
- Comparar los **tres escenarios de competencia** lado a lado.

## 🔮 Cómo funciona la predicción

El modelo predice **día por día de forma recursiva**: la predicción de cada día se usa como dato de entrada (lag) para predecir el siguiente.

Variables principales:

- **Historia reciente:** ventas de los 7 días anteriores (`lag_1` a `lag_7`) y su media móvil.
- **Precio:** precio de venta, descuento, precio de la competencia y ratio de precio contra la competencia.
- **Calendario:** día de la semana, fin de semana, inicio y fin de mes, festivos, Black Friday, Cyber Monday y Navidad.

## 📊 Modelo y resultados

| | |
|---|---|
| Modelo | _completar (por ejemplo, XGBoost, Random Forest…)_ |
| Métrica de evaluación | _completar (MAE, RMSE, MAPE…)_ |
| Resultado | _completar_ |
| Validación | _completar (por ejemplo, validación temporal)_ |

El entrenamiento y la evaluación están en los cuadernos de `notebooks/`.

## ⚠️ Limitaciones

- Al predecir de forma recursiva, **el error se va acumulando** a medida que avanza el mes.
- El simulador cambia el precio y el descuento, pero **la respuesta de las ventas es la que el modelo aprendió de datos históricos**: no prueba una relación causal.
- Los escenarios de competencia solo modifican el precio de los competidores; no modelan otras reacciones del mercado.

## 🛠️ Tecnologías

Python · Pandas · NumPy · _librería del modelo (completar)_ · Joblib · Streamlit · Jupyter

## 📁 Estructura del proyecto

```text
Forecastingventas/
├── app/            # app de Streamlit (app.py)
├── data/
│   ├── raw/        # datos originales, sin modificar
│   ├── processed/  # datos limpios y transformados
│   └── external/   # fuentes externas
├── models/         # modelo entrenado (modelo_final.joblib)
├── notebooks/      # exploración, preprocesamiento, entrenamiento y evaluación
├── src/            # código reutilizable
├── reports/figures/# gráficas exportadas
├── docs/           # documentación técnica
├── tests/          # pruebas automatizadas
└── requirements.txt
```

## ⚙️ Cómo ejecutarlo en tu computadora

```bash
python -m venv .venv
.venv\Scripts\activate        # en Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

Para reproducir el modelo: ejecutá los cuadernos de `notebooks/` en orden numérico y guardá el modelo en `models/`.

## 👤 Autor

**Fernando Javier Acosta**

- LinkedIn: https://www.linkedin.com/in/fernando-javier-acosta-153423289/
- Email: javierfer55555@gmail.com
