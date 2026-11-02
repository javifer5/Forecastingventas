"""Interfaz de demostracion para el pronostico de ventas."""

from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Forecasting de ventas", page_icon="📈", layout="wide")
st.title("Forecasting de ventas")
st.caption("Panel inicial para explorar datos y consultar pronosticos")

st.info(
    "Coloca un archivo CSV en data/raw/ para conectarlo al flujo de carga y prediccion."
)

raw_files = sorted((PROJECT_ROOT / "data" / "raw").glob("*.csv"))
if raw_files:
    st.success(f"{len(raw_files)} archivo(s) CSV encontrado(s) en data/raw.")
    st.dataframe([file.name for file in raw_files], use_container_width=True)
else:
    st.warning("Todavia no hay archivos CSV en data/raw.")
