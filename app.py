import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from datetime import datetime

# Configuración
st.set_page_config(page_title="Predicción de Papa", layout="wide")

# Leer modelos
@st.cache_resource
def cargar_modelos():
    return joblib.load("modelo_xgboost.pkl"), joblib.load("modelo_lightgbm.pkl")

modelo_xgb, modelo_lgb = cargar_modelos()

# Crear pestañas
pagina = st.sidebar.radio("Selecciona una opción", ["📈 Predicción", "📋 Historial"])

# ------------------ PREDICCIÓN ------------------
if pagina == "📈 Predicción":
    st.title("🌱 Predicción de Rendimiento de Cultivo de Papa")

    st.subheader("👤 Datos del Usuario")
    nombre = st.text_input("Nombre del Usuario")
    fecha = st.date_input("Fecha del Registro", value=datetime.today())
    ubicacion = st.text_input("Ubicación")

    st.subheader("🧪 Parámetros del Cultivo")

    # Opciones fijas
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

    if st.button("🔍 Predecir"):
        def codificar(v, lista): return lista.index(v)
        entrada = pd.DataFrame([[
            codificar(variedad, opciones_variedad),
            codificar(textura, opciones_textura),
            codificar(fertilizante, opciones_fertilizante),
            codificar(riego, opciones_riego),
            codificar(plagas, opciones_plagas),
            duracion, altitud, temperatura, precipitacion,
            ph, materia, dosis
        ]], columns=[
            'Variedad', 'Textura_Suelo', 'Uso_Fertilizante', 'Riego', 'Plagas',
            'Duración_Días', 'Altitud_msnm', 'Temperatura_Media_C',
            'Precipitación_mm', 'pH_Suelo', 'Materia_Orgánica_%', 'Dosis_Fertilizante_kg_ha'
        ])

        # Escalado
        scaler = joblib.load("scaler.pkl")
        entrada_scaled = scaler.transform(entrada)


        # Predicciones
        pred_xgb = modelo_xgb.predict(entrada_scaled)[0]
        pred_lgb = modelo_lgb.predict(entrada_scaled)[0]

        st.success(f"📈 Predicción XGBoost: **{pred_xgb:.2f} t/ha**")
        st.success(f"📈 Predicción LightGBM: **{pred_lgb:.2f} t/ha**")

        # Guardar en historial CSV
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
        st.success("✅ Registro guardado en historial")

# ------------------ HISTORIAL ------------------
elif pagina == "📋 Historial":
    st.title("📋 Historial de Predicciones")

    try:
        historial = pd.read_csv("historial_predicciones.csv")
        st.dataframe(historial)
    except FileNotFoundError:
        st.warning("⚠️ No hay registros aún. Realiza una predicción primero.")
