import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Configurar la página
st.set_page_config(page_title="Predicción de Papa", layout="wide")

# Cargar modelos y scaler
@st.cache_resource
def cargar_modelos():
    modelo_xgb = joblib.load("modelo_xgboost.pkl")
    modelo_lgb = joblib.load("modelo_lightgbm.pkl")
    scaler = joblib.load("scaler.pkl")
    return modelo_xgb, modelo_lgb, scaler

modelo_xgb, modelo_lgb, scaler = cargar_modelos()

# Pestañas (menú lateral)
pagina = st.sidebar.radio("Navegación", ["📈 Predicción", "📋 Historial"])

# ----------------------------------
# Página 1: Predicción manual
# ----------------------------------
if pagina == "📈 Predicción":
    st.title("🌱 Predicción de Rendimiento de Cultivo de Papa")

    st.subheader("👤 Datos del Usuario")
    nombre = st.text_input("Nombre del Usuario")
    fecha = st.date_input("Fecha del Registro", value=datetime.today())
    ubicacion = st.text_input("Ubicación o Región")

    st.subheader("🧪 Parámetros del Cultivo")

    # Opciones categóricas
    opciones_variedad = ['Desiree', 'Yungay', 'Canchan', 'Única', 'Perricholi']
    opciones_textura = ['Franco', 'Franco-arenoso', 'Franco-arcilloso']
    opciones_fertilizante = ['Sí', 'No']
    opciones_riego = ['Secano', 'Riego por goteo', 'Riego por aspersión']
    opciones_plagas = ['Baja', 'Media', 'Alta']

    col1, col2 = st.columns(2)
    with col1:
        variedad = st.selectbox("Variedad", opciones_variedad)
        textura = st.selectbox("Textura del Suelo", opciones_textura)
        fertilizante = st.selectbox("Uso de Fertilizante", opciones_fertilizante)
        riego = st.selectbox("Tipo de Riego", opciones_riego)
        plagas = st.selectbox("Nivel de Plagas", opciones_plagas)
        duracion = st.slider("Duración del cultivo (días)", 110, 150, 130)

    with col2:
        altitud = st.number_input("Altitud (msnm)", 2000, 4000, 3000)
        temperatura = st.slider("Temperatura media (°C)", 5.0, 30.0, 16.0)
        precipitacion = st.slider("Precipitación (mm)", 100, 700, 500)
        ph = st.slider("pH del Suelo", 4.5, 8.5, 6.5)
        materia = st.slider("Materia Orgánica (%)", 0.5, 5.0, 3.0)
        dosis = st.slider("Dosis Fertilizante (kg/ha)", 0, 300, 180)

    if st.button("🔍 Predecir rendimiento"):
        columnas_modelo = [
            'Variedad', 'Textura_Suelo', 'Uso_Fertilizante', 'Riego', 'Plagas',
            'Duración_Días', 'Altitud_msnm', 'Temperatura_Media_C',
            'Precipitación_mm', 'pH_Suelo', 'Materia_Orgánica_%', 'Dosis_Fertilizante_kg_ha'
        ]

        valores = [[
            opciones_variedad.index(variedad),
            opciones_textura.index(textura),
            opciones_fertilizante.index(fertilizante),
            opciones_riego.index(riego),
            opciones_plagas.index(plagas),
            duracion,
            altitud,
            temperatura,
            precipitacion,
            ph,
            materia,
            dosis
        ]]

        entrada_df = pd.DataFrame(valores, columns=columnas_modelo)
        entrada_scaled = scaler.transform(entrada_df)

        # Predicciones
        pred_xgb = modelo_xgb.predict(entrada_scaled)[0]
        pred_lgb = modelo_lgb.predict(entrada_scaled)[0]

        st.success(f"📈 Predicción XGBoost: **{pred_xgb:.2f} t/ha**")
        st.success(f"📈 Predicción LightGBM: **{pred_lgb:.2f} t/ha**")

        # Guardar en historial
        fila = {
            "Nombre": nombre,
            "Fecha": fecha,
            "Ubicación": ubicacion,
            "Variedad": variedad,
            "Textura_Suelo": textura,
            "Uso_Fertilizante": fertilizante,
            "Riego": riego,
            "Plagas": plagas,
            "Duración_Días": duracion,
            "Altitud_msnm": altitud,
            "Temperatura_Media_C": temperatura,
            "Precipitación_mm": precipitacion,
            "pH_Suelo": ph,
            "Materia_Orgánica_%": materia,
            "Dosis_Fertilizante_kg_ha": dosis,
            "Rendimiento_XGBoost": round(pred_xgb, 2),
            "Rendimiento_LightGBM": round(pred_lgb, 2)
        }

        try:
            historial = pd.read_csv("historial_predicciones.csv")
        except FileNotFoundError:
            historial = pd.DataFrame()

        historial = pd.concat([historial, pd.DataFrame([fila])], ignore_index=True)
        historial.to_csv("historial_predicciones.csv", index=False)
        st.success("✅ Registro guardado en historial.")

# ----------------------------------
# Página 2: Historial
# ----------------------------------
elif pagina == "📋 Historial":
    st.title("📋 Historial de Predicciones Registradas")
    try:
        historial = pd.read_csv("historial_predicciones.csv")
        st.dataframe(historial)
    except FileNotFoundError:
        st.warning("⚠️ No hay registros aún. Realiza una predicción primero.")
