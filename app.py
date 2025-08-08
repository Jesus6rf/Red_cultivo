import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="Predicción de Papa", layout="wide")

@st.cache_resource
def cargar_modelos():
    xgb = joblib.load("modelo_xgb.pkl")   # pipeline completo
    lgb = joblib.load("modelo_lgb.pkl")   # pipeline completo
    return xgb, lgb

xgb_pipe, lgb_pipe = cargar_modelos()

pagina = st.sidebar.radio("Navegación", ["📈 Predicción", "📋 Historial"])

if pagina == "📈 Predicción":
    st.title("🌱 Predicción de Rendimiento de Papa")

    # Datos del usuario
    nombre = st.text_input("Nombre")
    fecha = st.date_input("Fecha", value=datetime.today())
    ubicacion = st.text_input("Ubicación/Región")

    # Entradas (mismos nombres de columnas crudas del dataset)
    opciones_variedad = ['Desiree','Yungay','Canchan','Única','Perricholi']
    opciones_textura  = ['Franco','Franco-arenoso','Franco-arcilloso']
    opciones_fert     = ['Sí','No']
    opciones_riego    = ['Secano','Riego por goteo','Riego por aspersión']
    opciones_plagas   = ['Baja','Media','Alta']

    col1, col2 = st.columns(2)
    with col1:
        Variedad         = st.selectbox("Variedad", opciones_variedad)
        Textura_Suelo    = st.selectbox("Textura del Suelo", opciones_textura)
        Uso_Fertilizante = st.selectbox("Uso de Fertilizante", opciones_fert)
        Riego            = st.selectbox("Tipo de Riego", opciones_riego)
        Plagas           = st.selectbox("Nivel de Plagas", opciones_plagas)
        Duración_Días    = st.slider("Duración (días)", 110, 150, 130)
    with col2:
        Altitud_msnm           = st.number_input("Altitud (msnm)", 2000, 4000, 3000)
        Temperatura_Media_C    = st.slider("Temperatura media (°C)", 5.0, 30.0, 16.0)
        Precipitación_mm       = st.slider("Precipitación (mm)", 100, 700, 500)
        pH_Suelo               = st.slider("pH del Suelo", 4.5, 8.5, 6.5)
        Materia_Orgánica_pct   = st.slider("Materia Orgánica (%)", 0.5, 5.0, 3.0)
        Dosis_Fertilizante_kg_ha = st.slider("Dosis Fertilizante (kg/ha)", 0, 300, 180)

    if st.button("🔍 Predecir"):
        fila = pd.DataFrame([{
            'Variedad': Variedad,
            'Textura_Suelo': Textura_Suelo,
            'Uso_Fertilizante': Uso_Fertilizante,
            'Riego': Riego,
            'Plagas': Plagas,
            'Duración_Días': Duración_Días,
            'Altitud_msnm': Altitud_msnm,
            'Temperatura_Media_C': Temperatura_Media_C,
            'Precipitación_mm': Precipitación_mm,
            'pH_Suelo': pH_Suelo,
            'Materia_Orgánica_%': Materia_Orgánica_pct,
            'Dosis_Fertilizante_kg_ha': Dosis_Fertilizante_kg_ha
        }])

        pred_xgb = float(xgb_pipe.predict(fila)[0])
        pred_lgb = float(lgb_pipe.predict(fila)[0])

        st.success(f"🚀 XGBoost: **{pred_xgb:.2f} t/ha**")
        st.success(f"💡 LightGBM: **{pred_lgb:.2f} t/ha**")

        # Guardar historial
        fila_hist = fila.copy()
        fila_hist["Nombre"] = nombre
        fila_hist["Fecha"] = fecha
        fila_hist["Ubicación"] = ubicacion
        fila_hist["Pred_XGBoost_t_ha"] = round(pred_xgb, 2)
        fila_hist["Pred_LightGBM_t_ha"] = round(pred_lgb, 2)

        try:
            hist = pd.read_csv("historial_predicciones.csv")
        except FileNotFoundError:
            hist = pd.DataFrame()
        hist = pd.concat([hist, fila_hist], ignore_index=True)
        hist.to_csv("historial_predicciones.csv", index=False)
        st.success("✅ Registro guardado en historial.")

elif pagina == "📋 Historial":
    st.title("📋 Historial de Predicciones")
    try:
        hist = pd.read_csv("historial_predicciones.csv")
        st.dataframe(hist, use_container_width=True)
    except FileNotFoundError:
        st.info("Aún no hay registros.")
