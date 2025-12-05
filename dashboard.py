# Ejecutar con:
# python -m streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np

# Parche para compatibilidad con NumPy 2.0
if not hasattr(np, "float_"):
    np.float_ = np.float64

import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# =========================================================
# CONFIGURACIÓN DE PÁGINA
# =========================================================
st.set_page_config(
    page_title="Bepensa Logistics - Dashboard Predictivo",
    layout="wide",
)

# =========================================================
# CARGA DE DF_maestro DESDE CSV
# =========================================================
@st.cache_data
def cargar_df_maestro():
    """
    Lee DF_maestro desde un CSV.
    Asegúrate de exportar desde el notebook un archivo DF_maestro.csv
    con las columnas:
        - fecha
        - region_x
        - km_total_vehiculo
        - num_fallas
    """
    df = pd.read_csv("DF_maestro.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df

try:
    DF_maestro = cargar_df_maestro()
except FileNotFoundError:
    st.error(
        "No se encontró 'DF_maestro.csv'. "
        "Expórtalo desde tu notebook y colócalo en la misma carpeta que app.py."
    )
    st.stop()

# =========================================================
# BLOQUE COSTOS DE BUFFER (a partir de Combustible.xlsx - Hoja1.csv)
# =========================================================
@st.cache_data
def calcular_costo_km_buffer():
    """
    Calcula el costo promedio de combustible por km (MXN/km)
    usando la base 'Combustible.xlsx - Hoja1.csv', replicando tu lógica.
    """
    combustible_costos = pd.read_csv("Combustible.xlsx - Hoja1.csv", low_memory=False)

    combustible_costos["Recorrido"] = pd.to_numeric(
        combustible_costos["Recorrido"], errors="coerce"
    )
    combustible_costos["Importe Transacción"] = pd.to_numeric(
        combustible_costos["Importe Transacción"], errors="coerce"
    )

    mask_valid = (
        (combustible_costos["Recorrido"] >= 5)
        & (combustible_costos["Recorrido"] <= 3000)
        & (combustible_costos["Importe Transacción"] > 0)
    )

    combustible_filtrado = combustible_costos[mask_valid].copy()

    costo_km_buffer = (
        combustible_filtrado["Importe Transacción"].sum()
        / combustible_filtrado["Recorrido"].sum()
    )

    return float(costo_km_buffer)

# Costo de subcontratación por km (tu referencia)
COSTO_SUBCONTRATACION_KM = 40.0  # MXN/km

try:
    COSTO_KM_BUFFER = calcular_costo_km_buffer()
except FileNotFoundError:
    st.error(
        "No se encontró 'Combustible.xlsx - Hoja1.csv'. "
        "Colócalo en la misma carpeta que app.py o ajusta la ruta."
    )
    st.stop()

# =========================================================
# FUNCIÓN: PIPELINE POR REGIÓN (basado en tu código)
# =========================================================
FACTOR_PENALIZACION_RIESGO = 0.20

@st.cache_data
def correr_pipeline_region(df_maestro, region):
    """
    Reproduce la lógica de tu loop por región, pero regresando
    todo lo necesario para graficar en Plotly.
    """

    df_region = df_maestro[df_maestro["region_x"] == region].copy()

    ts_data = df_region.groupby("fecha").agg({
        "km_total_vehiculo": "sum",
        "num_fallas": "sum"
    }).reset_index()

    ts_data = ts_data.sort_values("fecha")

    df_prophet = ts_data[["fecha", "km_total_vehiculo"]].rename(
        columns={"fecha": "ds", "km_total_vehiculo": "y"}
    )

    # Cap y floor
    cap_value = df_prophet["y"].max() * 1.1
    floor_value = df_prophet["y"].min() * 0.9
    df_prophet["cap"] = cap_value
    df_prophet["floor"] = floor_value

    # Validación 80/20
    if len(df_prophet) < 15:
        return {
            "ok": False,
            "mensaje": "Datos insuficientes para validación temporal (menos de 15 puntos)."
        }

    corte_index = int(len(df_prophet) * 0.80)
    df_train = df_prophet.iloc[:corte_index].copy()
    df_test = df_prophet.iloc[corte_index:].copy()

    df_train["cap"] = cap_value
    df_train["floor"] = floor_value

    m = Prophet(
        growth="logistic",
        daily_seasonality=False,
        weekly_seasonality=True
    )
    m.fit(df_train)

    future_test_df = df_test[["ds", "cap", "floor"]].copy()
    forecast_test = m.predict(future_test_df)

    y_true = df_test["y"].values
    y_pred = forecast_test["yhat"].values

    if len(y_true) > 0:
        smape = np.mean(
            2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
        )
        mae = mean_absolute_error(y_true, y_pred)
    else:
        smape = np.nan
        mae = np.nan

    # Entrenamiento final 100%
    m_final = Prophet(
        growth="logistic",
        daily_seasonality=False,
        weekly_seasonality=True
    )
    m_final.fit(df_prophet)

    future_final = m_final.make_future_dataframe(periods=7)
    future_final["cap"] = cap_value
    future_final["floor"] = floor_value

    forecast_final = m_final.predict(future_final)

    demanda_proyectada = forecast_final.tail(7)["yhat"].sum()

    # Z-score de fallas
    media_fallas = ts_data["num_fallas"].mean()
    std_fallas = ts_data["num_fallas"].std() if ts_data["num_fallas"].std() > 0 else 1

    fallas_actuales = ts_data["num_fallas"].iloc[-1]
    z_score_fallas = (fallas_actuales - media_fallas) / std_fallas

    capacidad_maxima = ts_data["km_total_vehiculo"].max()
    capacidad_efectiva = capacidad_maxima
    estado_flota = "ESTABLE"

    if z_score_fallas > 1:
        estado_flota = "CRÍTICO"
        capacidad_efectiva = capacidad_maxima * (1 - FACTOR_PENALIZACION_RIESGO)

    gap = demanda_proyectada - capacidad_efectiva

    return {
        "ok": True,
        "df_prophet": df_prophet,
        "df_test": df_test.reset_index(drop=True),
        "forecast_test": forecast_test.reset_index(drop=True),
        "forecast_final": forecast_final,
        "ts_data": ts_data,
        "smape": smape,
        "mae": mae,
        "demanda_proyectada": demanda_proyectada,
        "capacidad_efectiva": capacidad_efectiva,
        "media_fallas": media_fallas,
        "std_fallas": std_fallas,
        "fallas_actuales": fallas_actuales,
        "z_score_fallas": z_score_fallas,
        "gap": gap,
        "estado_flota": estado_flota,
    }

# =========================================================
# RESUMEN POR REGIÓN PARA ÓRDENES
# =========================================================
@st.cache_data
def resumen_buffering_por_region(df_maestro):
    regiones = df_maestro["region_x"].dropna().unique()
    filas = []

    for region in regiones:
        res = correr_pipeline_region(df_maestro, region)
        if not res["ok"]:
            continue

        demanda = res["demanda_proyectada"]
        capacidad = res["capacidad_efectiva"]
        gap = res["gap"]
        estado = res["estado_flota"]
        z = res["z_score_fallas"]

        km_buffer = max(gap, 0)
        costo_pre = km_buffer * COSTO_KM_BUFFER
        costo_int = demanda * COSTO_KM_BUFFER
        costo_sub = demanda * COSTO_SUBCONTRATACION_KM

        if demanda > 0:
            valor_buffer = (costo_sub - (costo_pre + costo_int)) / demanda
        else:
            valor_buffer = 0.0

        requiere = "SÍ" if gap > 0 else "NO"

        filas.append({
            "Región": region,
            "Demanda Proyectada (Km)": round(demanda, 0),
            "Capacidad Efectiva (Km)": round(capacidad, 0),
            "Gap (Km)": round(gap, 0),
            "Estado Flota": f"{estado} (Z={z:.2f})",
            "Km Buffer Necesario": round(km_buffer, 0),
            "Costo Operación Interna (MXN)": round(costo_int, 2),
            "Costo Preposicionamiento (MXN)": round(costo_pre, 2),
            "Costo Subcontratación (MXN)": round(costo_sub, 2),
            "Valor Buffer (MXN/km)": round(valor_buffer, 2),
            "Requiere Buffering": requiere,
        })

    return pd.DataFrame(filas)

# =========================================================
# HEADER (solo texto)
# =========================================================
header_col1, header_col2 = st.columns([1, 3])

with header_col1:
    st.markdown(
        """
        <h2 style="margin-bottom:0;">Bepensa</h2>
        <p style="margin-top:0;color:#f58220;font-weight:600;">LOGISTICS</p>
        """,
        unsafe_allow_html=True,
    )

with header_col2:
    st.markdown(
        """
        <h2 style="text-align:right;color:#444;margin-bottom:0;">
        Prototipo de Dashboard Predictivo y Buffering
        </h2>
        <p style="text-align:right;color:#777;margin-top:4px;">
        Modelo de predicción de kilómetros, riesgo de fallas y costos de buffering por región
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================================================
# MENÚ PRINCIPAL
# =========================================================
pagina = st.radio(
    "Navegación",
    options=["Inicio", "Dashboard", "Órdenes"],
    horizontal=True,
    label_visibility="collapsed",
)

# =========================================================
# PÁGINA: INICIO
# =========================================================
if pagina == "Inicio":
    st.subheader("LA EFICIENCIA EMPIEZA CON VISIBILIDAD")
    st.write(
        f"""
        Este tablero integra:
        
        - Predicción de kilómetros recorridos por región usando **Prophet**.
        - Cálculo de capacidad efectiva y **gap operativo**.
        - Evaluación del riesgo de fallas mediante **Z-score**.
        - Estimación de **costos de buffer**, usando un costo promedio de combustible
          de **{COSTO_KM_BUFFER:.2f} MXN/km** y una tarifa de subcontratación de
          **{COSTO_SUBCONTRATACION_KM:.2f} MXN/km**.
        """
    )

# =========================================================
# PÁGINA: DASHBOARD  (GRÁFICAS + COSTOS DE BUFFER)
# =========================================================
elif pagina == "Dashboard":
    st.subheader("Dashboard de predicción, riesgo y costos de buffering")

    st.markdown("### Selecciona la región para analizar el pipeline completo")

    regiones = DF_maestro["region_x"].dropna().unique()
    region_sel = st.selectbox("Región", sorted(regiones))

    resultados = correr_pipeline_region(DF_maestro, region_sel)

    if not resultados["ok"]:
        st.warning(resultados["mensaje"])
        st.stop()

    # Extraemos variables
    df_prophet = resultados["df_prophet"]
    df_test = resultados["df_test"]
    forecast_test = resultados["forecast_test"]
    forecast_final = resultados["forecast_final"]
    ts_data = resultados["ts_data"]
    # smape = resultados["smape"]  # si lo quieres luego, aquí está
    # mae = resultados["mae"]
    demanda_proyectada = resultados["demanda_proyectada"]
    capacidad_efectiva = resultados["capacidad_efectiva"]
    media_fallas = resultados["media_fallas"]
    std_fallas = resultados["std_fallas"]
    fallas_actuales = resultados["fallas_actuales"]
    z_score_fallas = resultados["z_score_fallas"]
    gap = resultados["gap"]
    estado_flota = resultados["estado_flota"]

    # =====================================================
    # CÁLCULO DE COSTOS DE BUFFER PARA LA REGIÓN SELECCIONADA
    # =====================================================
    km_buffer_necesario = max(gap, 0)  # si hay déficit, es el km de buffer requerido

    costo_km_buffer = COSTO_KM_BUFFER

    costo_preposicionamiento = km_buffer_necesario * costo_km_buffer
    costo_operacion_interna = demanda_proyectada * costo_km_buffer
    costo_subcontratacion_total = demanda_proyectada * COSTO_SUBCONTRATACION_KM

    if demanda_proyectada > 0:
        valor_buffer_unitario = (
            costo_subcontratacion_total
            - (costo_preposicionamiento + costo_operacion_interna)
        ) / demanda_proyectada
    else:
        valor_buffer_unitario = 0.0

    # ===================== KPIs DE COSTO ==================
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("Estado flota", f"{estado_flota}", help=f"Z-score fallas: {z_score_fallas:.2f}")
    with k2:
        st.metric("Km de buffer necesario", f"{km_buffer_necesario:,.0f} km")
    with k3:
        st.metric("Costo por km de buffer", f"{costo_km_buffer:,.2f} MXN/km")
    with k4:
        st.metric("Valor del buffer", f"{valor_buffer_unitario:,.2f} MXN/km")

    st.caption(
        "El valor del buffer compara el costo de subcontratar toda la demanda vs operar "
        "internamente + preposicionar km de buffer."
    )

    # ---------------------- GRÁFICA 1: VALIDACIÓN ----------------------
    st.markdown("#### 1. Validación: Forecast vs Real (20% Test)")

    df_val = pd.DataFrame({
        "ds": df_test["ds"],
        "Real (20%)": df_test["y"],
        "Forecast (20%)": forecast_test["yhat"]
    })

    df_val_long = df_val.melt(id_vars="ds", var_name="serie", value_name="km")

    fig1 = px.line(
        df_val_long,
        x="ds",
        y="km",
        color="serie",
        markers=True,
        title=f"VALIDACIÓN — Región {region_sel}"
    )
    fig1.update_layout(xaxis_title="Fecha", yaxis_title="Km Totales")

    # ---------------------- GRÁFICA 2: FORECAST FINAL ----------------------
    st.markdown("#### 2. Predicción operativa: Forecast 7 días")

    df_hist = df_prophet[["ds", "y"]].copy()
    df_hist.rename(columns={"y": "km_hist"}, inplace=True)

    forecast7 = forecast_final.tail(7).copy()

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatter(
            x=df_hist["ds"],
            y=df_hist["km_hist"],
            mode="lines",
            name="Histórico",
            line=dict(color="blue")
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=forecast7["ds"],
            y=forecast7["yhat"],
            mode="lines+markers",
            name="Forecast 7 días",
            line=dict(color="red", width=3)
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=list(forecast7["ds"]) + list(forecast7["ds"][::-1]),
            y=list(forecast7["yhat_upper"]) + list(forecast7["yhat_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(255,0,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Intervalo de confianza"
        )
    )

    fig2.update_layout(
        title=f"PREDICCIÓN OPERATIVA: Forecast 7 Días — Región {region_sel}",
        xaxis_title="Fecha",
        yaxis_title="Km Totales"
    )

    # ---------------------- GRÁFICA 3: FALLAS Y Z-SCORE ----------------------
    st.markdown("#### 3. Monitoreo de riesgo: Fallas y umbral Z-score")

    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatter(
            x=ts_data["fecha"],
            y=ts_data["num_fallas"],
            mode="lines+markers",
            name="Fallas registradas",
            line=dict(color="royalblue")
        )
    )

    fig3.add_hline(
        y=media_fallas,
        line_dash="dash",
        line_color="orange",
        annotation_text="Media de fallas",
        annotation_position="top left"
    )

    fig3.add_hline(
        y=media_fallas + std_fallas,
        line_dash="dot",
        line_color="red",
        annotation_text="Z-score > 1 (Riesgo)",
        annotation_position="top left"
    )

    fig3.add_trace(
        go.Scatter(
            x=[ts_data["fecha"].iloc[-1]],
            y=[fallas_actuales],
            mode="markers",
            name=f"Falla actual ({fallas_actuales})",
            marker=dict(color="black", size=10)
        )
    )

    fig3.update_layout(
        title=f"MONITOREO DE RIESGO: Fallas y Umbral Z-Score — Región {region_sel}",
        xaxis_title="Fecha",
        yaxis_title="Número de fallas"
    )

    # ---------------------- GRÁFICA 4: GAP Y BUFFERING ----------------------
    st.markdown("#### 4. Gap operativo y km de buffer")

    df_gap_costos = pd.DataFrame({
        "Concepto": [
            "Demanda Proyectada (Km)",
            "Capacidad Efectiva (Km)",
            "Km Buffer Necesario",
        ],
        "Km": [
            demanda_proyectada,
            capacidad_efectiva,
            km_buffer_necesario,
        ],
    })

    fig4a = px.bar(
        df_gap_costos,
        x="Concepto",
        y="Km",
        title=f"Gap Operativo — Región {region_sel}",
        color="Concepto",
        color_discrete_sequence=["steelblue", "seagreen", "red"],
    )
    fig4a.update_layout(yaxis_title="Kilometraje (Km)")

    # GRID 2x2
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    with r1c1:
        st.plotly_chart(fig1, use_container_width=True)
    with r1c2:
        st.plotly_chart(fig2, use_container_width=True)
    with r2c1:
        st.plotly_chart(fig3, use_container_width=True)
    with r2c2:
        st.plotly_chart(fig4a, use_container_width=True)

# =========================================================
# PÁGINA: ÓRDENES (usa la info de buffering)
# =========================================================
elif pagina == "Órdenes":
    st.subheader("Órdenes y recomendación de buffering por región")

    st.write(
        "Este panel resume, por región, la demanda proyectada, el estado de la flota, "
        "los kilómetros de buffer sugeridos y el costo asociado al buffering."
    )

    df_resumen = resumen_buffering_por_region(DF_maestro)

    if df_resumen.empty:
        st.warning("No se pudo generar el resumen de buffering (posiblemente hay muy pocos datos).")
    else:
        st.markdown("### Resumen de decisión de buffering por región")
        st.dataframe(df_resumen, use_container_width=True)

        # ❌ SE ELIMINA ESTA GRÁFICA
        # st.markdown("### Km de buffer necesario por región")
        # fig_ord = px.bar(
        #     df_resumen,
        #     x="Región",
        #     y="Km Buffer Necesario",
        #     color="Requiere Buffering",
        #     title="Requerimiento de km de buffer por región",
        #     color_discrete_map={"SÍ": "red", "NO": "seagreen"},
        # )
        # fig_ord.update_layout(yaxis_title="Km de Buffer Necesario")
        # st.plotly_chart(fig_ord, use_container_width=True)
    