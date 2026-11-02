from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import joblib


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Forecasting de Ventas - Noviembre 2025",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# ESTILO VISUAL
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f8fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #4c51bf;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #4c51bf;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    .scenario-card {
        padding: 1rem;
        border-radius: 12px;
        background-color: white;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .main h3 {
        color: #1f2937 !important;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        color: #1f2937 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #1f2937 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent        # raíz del repo
RUTA_DATOS = BASE_DIR / "data" / "processed" / "inferencia_df_transformado.csv"
RUTA_MODELO = BASE_DIR / "models" / "modelo_final.joblib"

COLUMNAS_LAG = [
    "unidades_vendidas_lag_1",
    "unidades_vendidas_lag_2",
    "unidades_vendidas_lag_3",
    "unidades_vendidas_lag_4",
    "unidades_vendidas_lag_5",
    "unidades_vendidas_lag_6",
    "unidades_vendidas_lag_7"
]

COLUMNAS_COMPETENCIA = [
    "Amazon",
    "Decathlon",
    "Deporvillage"
]


# ============================================================
# CARGA DE DATOS Y MODELO
# ============================================================

@st.cache_data
def cargar_datos():

    if not RUTA_DATOS.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {RUTA_DATOS}"
        )

    return pd.read_csv(RUTA_DATOS)


@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


try:
    df = cargar_datos()
except Exception as error:
    st.error(
        "❌ No se pudo cargar el archivo "
        "`data/processed/inferencia_df_transformado.csv`."
    )
    st.exception(error)
    st.stop()


try:
    modelo = cargar_modelo()
except Exception as error:
    st.error(
        "❌ No se pudo cargar el modelo "
        "`models/modelo_final.joblib`."
    )
    st.exception(error)
    st.stop()


# ============================================================
# VALIDACIÓN DE COLUMNAS
# ============================================================

COLUMNAS_OBLIGATORIAS = [
    "fecha",
    "producto_id",
    "nombre",
    "categoria",
    "subcategoria",
    "precio_base",
    "precio_venta",
    "año",
    "mes",
    "trimestre",
    "semana_año",
    "dia_año",
    "dia_mes",
    "dia_semana",
    "dia_semana_numero",
    "es_fin_semana",
    "es_inicio_semana",
    "es_viernes",
    "es_inicio_mes",
    "es_fin_mes",
    "es_inicio_trimestre",
    "es_fin_trimestre",
    "es_festivo",
    "nombre_festivo",
    "es_black_friday",
    "es_cyber_monday",
    "es_navidad",
    "es_nochebuena",
    "es_reyes",
    "dias_desde_inicio_año",
    "unidades_vendidas_lag_1",
    "unidades_vendidas_lag_2",
    "unidades_vendidas_lag_3",
    "unidades_vendidas_lag_4",
    "unidades_vendidas_lag_5",
    "unidades_vendidas_lag_6",
    "unidades_vendidas_lag_7",
    "media_movil_7_dias",
    "descuento_porcentaje",
    "precio_competencia",
    "ratio_precio"
]


COLUMNAS_FALTANTES = [
    columna
    for columna in COLUMNAS_OBLIGATORIAS
    if columna not in df.columns
]


if COLUMNAS_FALTANTES:
    st.error("❌ Faltan columnas obligatorias en el dataframe:")
    st.write(COLUMNAS_FALTANTES)
    st.stop()


if not hasattr(modelo, "feature_names_in_"):
    st.error(
        "❌ El modelo no contiene `feature_names_in_`. "
        "No es posible garantizar que las columnas utilizadas "
        "sean exactamente las que espera el modelo."
    )
    st.stop()


FEATURES_MODELO = list(modelo.feature_names_in_)

COLUMNAS_MODELO_FALTANTES = [
    columna
    for columna in FEATURES_MODELO
    if columna not in df.columns
]


if COLUMNAS_MODELO_FALTANTES:
    st.error(
        "❌ El dataframe no contiene todas las columnas que espera "
        "el modelo."
    )
    st.write(COLUMNAS_MODELO_FALTANTES)
    st.stop()


# ============================================================
# PREPARACIÓN DE FECHAS
# ============================================================

df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

if df["fecha"].isna().any():
    st.error("❌ Hay fechas inválidas en el dataframe.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("# 🎛️ Controles de Simulación")

st.sidebar.markdown(
    "Selecciona el producto y modifica las condiciones "
    "de la simulación."
)

productos = sorted(
    df["nombre"].dropna().unique().tolist()
)

if not productos:
    st.error("❌ No se encontraron productos en el dataframe.")
    st.stop()


producto_seleccionado = st.sidebar.selectbox(
    "📦 Producto",
    productos
)


ajuste_descuento = st.sidebar.slider(
    "🏷️ Ajuste de descuento",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    format="%d%%"
)


escenario_competencia = st.sidebar.radio(
    "🏪 Escenario de competencia",
    [
        "Actual (0%)",
        "Competencia -5%",
        "Competencia +5%"
    ]
)


simular = st.sidebar.button(
    "🚀 Simular Ventas",
    use_container_width=True,
    type="primary"
)


# ============================================================
# FUNCIONES
# ============================================================

def calcular_precio_competencia(
    fila,
    escenario
):
    """
    Calcula el precio promedio de competencia.

    Las columnas Amazon, Decathlon y Deporvillage
    representan los precios individuales de los competidores.
    """

    precios = []

    for columna in COLUMNAS_COMPETENCIA:

        if columna in fila.index:

            precio = float(fila[columna])

            if escenario == "Competencia -5%":
                precio = precio * 0.95

            elif escenario == "Competencia +5%":
                precio = precio * 1.05

            precios.append(precio)

    if not precios:

        return float(fila["precio_competencia"])

    return float(np.mean(precios))


def preparar_dataframe_producto(
    dataframe,
    producto,
    descuento,
    escenario
):
    """
    Filtra el producto y prepara únicamente las variables
    dependientes de los controles de simulación.
    """

    datos = dataframe[
        dataframe["nombre"] == producto
    ].copy()

    datos = datos.sort_values("fecha").reset_index(drop=True)

    if datos.empty:
        return datos

    # Precio de venta a partir del precio base.
    datos["descuento_porcentaje"] = descuento

    datos["precio_venta"] = (
        datos["precio_base"]
        * (1 - descuento / 100)
    )

    # Precio competencia.
    precios_competencia = []

    for indice in range(len(datos)):

        fila = datos.iloc[indice]

        precio_competencia = calcular_precio_competencia(
            fila,
            escenario
        )

        precios_competencia.append(
            precio_competencia
        )

    datos["precio_competencia"] = precios_competencia

    # Ratio precio.
    datos["ratio_precio"] = np.where(
        datos["precio_competencia"] != 0,
        datos["precio_venta"]
        / datos["precio_competencia"],
        0
    )

    return datos


def actualizar_lags(
    datos,
    indice,
    predicciones
):
    """
    Actualiza los lags para el día actual utilizando
    las predicciones anteriores.

    Para el día 1 no modifica los lags originales.
    """

    if indice == 0:
        return datos

    prediccion_anterior = predicciones[-1]
    anterior = indice - 1

    # Los lags de hoy se arman desde los lags YA ACTUALIZADOS de ayer:
    #   lag_1(hoy) = predicción de ayer
    #   lag_k(hoy) = lag_(k-1) de ayer, para k = 2..7
    nuevos_lags = [prediccion_anterior]

    for numero_lag in range(2, 8):
        nuevos_lags.append(
            float(datos.loc[anterior, "unidades_vendidas_lag_" + str(numero_lag - 1)])
        )

    for columna, valor in zip(COLUMNAS_LAG, nuevos_lags):
        datos.loc[indice, columna] = valor

    # ========================================================
    # MEDIA MÓVIL RECURSIVA
    # ========================================================

    # Los 7 lags representan la historia disponible
    # inmediatamente anterior al día que estamos prediciendo.
    #
    # Para el día 2, lag_1 contiene la predicción del día 1
    # y los demás lags conservan la historia inicial.
    #
    # Para cada día posterior se vuelve a calcular utilizando
    # los siete valores disponibles.

    valores_historial = []

    for columna in COLUMNAS_LAG:
        valores_historial.append(
            float(datos.loc[indice, columna])
        )

    datos.loc[
        indice,
        "media_movil_7_dias"
    ] = np.mean(valores_historial)

    return datos


def predecir_recursivamente(
    datos,
    modelo
):
    """
    Realiza las predicciones día por día.

    Día 1:
        utiliza los lags originales del CSV.

    Días siguientes:
        actualiza los lags con las predicciones anteriores.
    """

    datos = datos.copy()

    datos = datos.sort_values(
        "fecha"
    ).reset_index(drop=True)

    predicciones = []

    with st.spinner(
        "🔮 Ejecutando predicciones recursivas día por día..."
    ):

        for indice in range(len(datos)):

            # Para día 1 no se modifican los lags.
            # Para los siguientes se actualizan.
            if indice > 0:
                datos = actualizar_lags(
                    datos,
                    indice,
                    predicciones
                )

            fila = datos.iloc[[indice]].copy()

            # Se utiliza exclusivamente el conjunto de
            # features que el modelo espera.
            fila_modelo = fila[
                FEATURES_MODELO
            ].copy()

            try:
                prediccion = modelo.predict(
                    fila_modelo
                )[0]

            except Exception as error:

                raise RuntimeError(
                    "Error al predecir el día "
                    + str(indice + 1)
                    + ": "
                    + str(error)
                )

            # Las unidades vendidas no pueden ser negativas.
            prediccion = max(
                0,
                float(prediccion)
            )

            predicciones.append(
                prediccion
            )

    datos["unidades_predichas"] = (
        predicciones
    )

    datos["ingresos_proyectados"] = (
        datos["unidades_predichas"]
        * datos["precio_venta"]
    )

    return datos


def ejecutar_simulacion(
    producto,
    descuento,
    escenario
):

    datos = preparar_dataframe_producto(
        df,
        producto,
        descuento,
        escenario
    )

    if datos.empty:
        raise ValueError(
            "No existen datos para el producto seleccionado."
        )

    return predecir_recursivamente(
        datos,
        modelo
    )


def ejecutar_escenarios(
    producto,
    descuento
):

    resultados = {}

    escenarios = [
        "Actual (0%)",
        "Competencia -5%",
        "Competencia +5%"
    ]

    with st.spinner(
        "📊 Calculando los tres escenarios de competencia..."
    ):

        for escenario in escenarios:

            resultados[escenario] = ejecutar_simulacion(
                producto,
                descuento,
                escenario
            )

    return resultados


# ============================================================
# FUNCIÓN PRINCIPAL DE DASHBOARD
# ============================================================

def mostrar_dashboard(
    resultados,
    producto,
    descuento,
    escenario
):

    datos = resultados[escenario]

    # ========================================================
    # HEADER
    # ========================================================

    st.title("📈 Forecasting de Ventas")
    st.subheader(
        "Simulación de ventas — Noviembre 2025 | "
        + str(producto)
    )

    st.info(
        "ℹ️ La simulación utiliza predicciones recursivas: "
        "las ventas predichas de cada día alimentan los lags "
        "de los días posteriores."
    )

    st.divider()

    # ========================================================
    # KPIs
    # ========================================================

    unidades_totales = (
        datos["unidades_predichas"].sum()
    )

    ingresos_totales = (
        datos["ingresos_proyectados"].sum()
    )

    precio_promedio = (
        datos["precio_venta"].mean()
    )

    descuento_promedio = (
        datos["descuento_porcentaje"].mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📦 Unidades proyectadas",
            f"{unidades_totales:,.0f}"
        )

    with col2:
        st.metric(
            "💰 Ingresos proyectados",
            f"€ {ingresos_totales:,.2f}"
        )

    with col3:
        st.metric(
            "🏷️ Precio promedio",
            f"€ {precio_promedio:,.2f}"
        )

    with col4:
        st.metric(
            "🔖 Descuento promedio",
            f"{descuento_promedio:,.0f}%"
        )

    st.divider()

    # ========================================================
    # GRÁFICO
    # ========================================================

    st.header("📊 Predicción diaria de ventas")

    grafico = datos.groupby(
        "dia_mes",
        as_index=False
    )["unidades_predichas"].mean().rename(
        columns={
            "dia_mes": "Día de noviembre",
            "unidades_predichas": "Unidades predichas"
        }
    )

    st.line_chart(
        grafico,
        x="Día de noviembre",
        y="Unidades predichas",
        x_label="Día de noviembre",
        y_label="Unidades predichas",
        height=400
    )

    # Black Friday.
    black_friday = datos[
        datos["dia_mes"] == 28
    ]

    if not black_friday.empty:

        valor_black_friday = float(
            black_friday[
                "unidades_predichas"
            ].iloc[0]
        )

        st.caption(
            "🛍️ Black Friday (28/11): "
            + f"{valor_black_friday:,.0f} unidades predichas."
        )

    st.divider()

    # ========================================================
    # TABLA
    # ========================================================

    st.header("📋 Detalle diario")

    tabla = datos.copy()

    tabla["fecha"] = (
        tabla["fecha"].dt.strftime("%d/%m/%Y")
    )

    tabla["precio_venta"] = (
        tabla["precio_venta"]
        .round(2)
    )

    tabla["precio_competencia"] = (
        tabla["precio_competencia"]
        .round(2)
    )

    tabla["descuento_porcentaje"] = (
        tabla["descuento_porcentaje"]
        .round(0)
    )

    tabla["unidades_predichas"] = (
        tabla["unidades_predichas"]
        .round(0)
        .astype(int)
    )

    tabla["ingresos_proyectados"] = (
        tabla["ingresos_proyectados"]
        .round(2)
    )

    tabla["dia_semana"] = (
        tabla["dia_semana"]
        .astype(str)
    )

    tabla = tabla[
        [
            "fecha",
            "dia_semana",
            "precio_venta",
            "precio_competencia",
            "descuento_porcentaje",
            "unidades_predichas",
            "ingresos_proyectados"
        ]
    ].copy()

    tabla.columns = [
        "Fecha",
        "Día de semana",
        "Precio venta (€)",
        "Precio competencia (€)",
        "Descuento (%)",
        "Unidades predichas",
        "Ingresos proyectados (€)"
    ]

    def destacar_black_friday(
        fila
    ):
        if fila["Fecha"] == "28/11/2025":
            return [
                "background-color: #ffe4e6"
            ] * len(fila)

        return [
            ""
        ] * len(fila)

    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ========================================================
    # COMPARATIVA DE ESCENARIOS
    # ========================================================

    st.header("🏪 Comparativa de escenarios")

    st.caption(
        "El descuento seleccionado se mantiene constante. "
        "Solo cambia el precio de la competencia."
    )

    col1, col2, col3 = st.columns(3)

    nombres_escenarios = [
        "Actual (0%)",
        "Competencia -5%",
        "Competencia +5%"
    ]

    columnas = [
        col1,
        col2,
        col3
    ]

    iconos = [
        "⚪",
        "🟢",
        "🔴"
    ]

    for columna, nombre, icono in zip(
        columnas,
        nombres_escenarios,
        iconos
    ):

        datos_escenario = resultados[
            nombre
        ]

        unidades = (
            datos_escenario[
                "unidades_predichas"
            ].sum()
        )

        ingresos = (
            datos_escenario[
                "ingresos_proyectados"
            ].sum()
        )

        with columna:
            with st.container(border=True):
                st.subheader(
                    icono + " " + nombre,
                    anchor=False
                )

                st.metric(
                    "Unidades",
                    f"{unidades:,.0f}"
                )

                st.metric(
                    "Ingresos",
                    f"€ {ingresos:,.2f}"
                )


# ============================================================
# EJECUCIÓN
# ============================================================

if simular:

    try:

        with st.spinner(
            "🚀 Preparando simulación..."
        ):

            resultados = ejecutar_escenarios(
                producto_seleccionado,
                ajuste_descuento
            )

        mostrar_dashboard(
            resultados,
            producto_seleccionado,
            ajuste_descuento,
            escenario_competencia
        )

    except Exception as error:

        st.error(
            "❌ Se produjo un error durante la simulación."
        )

        st.exception(error)

else:

    st.title("📈 Forecasting de Ventas")
    st.subheader("Simulador de ventas para noviembre 2025")

    st.info(
        "👈 Selecciona un producto, ajusta el descuento "
        "y el escenario de competencia en el panel lateral. "
        "Después pulsa **🚀 Simular Ventas**."
    )

    st.markdown(
        "### 🎯 ¿Qué hace esta aplicación?"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "**🔮 Predicción recursiva**\n\n"
            "Cada predicción diaria alimenta los lags "
            "de los días siguientes."
        )

    with col2:
        st.markdown(
            "**🏷️ Simulación comercial**\n\n"
            "Puedes modificar el descuento y el escenario "
            "de competencia."
        )

    with col3:
        st.markdown(
            "**📊 Dashboard**\n\n"
            "Visualiza unidades, ingresos, precios y "
            "la evolución diaria."
        )