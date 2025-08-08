import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Predicción de Rendimiento de Papa", layout="centered")
st.title("🌱 Predicción Manual del Rendimiento de Cultivo de Papa")

# Cargar modelos
@st.cache_resource
def cargar_modelos():
    modelo_xgb = joblib.load("modelo_xgboost.pkl")
    modelo_lgb = joblib.load("modelo_lightgbm.pkl")
    return modelo_xgb, modelo_lgb

modelo_xgb, modelo_lgb = cargar_modelos()

# Opciones fijas (deben coincidir con las usadas en entrenamiento)
opciones_variedad = ['Desiree', 'Yungay', 'Canchan', 'Única', 'Perricholi']
opciones_textura = ['Franco', 'Franco-arenoso', 'Franco-arcilloso']
opciones_fertilizante = ['Sí', 'No']
opciones_riego = ['Secano', 'Riego por goteo', 'Riego por aspersión']
opciones_plagas = ['Baja', 'Media', 'Alta']

# Inputs del usuario
st.subheader("🔧 Ingresa los datos del cultivo:")

col1, col2 = st.columns(2)

with col1:
    variedad = st.selectbox("Variedad", opciones_variedad)
    textura = st.selectbox("Textura del Suelo", opciones_textura)
    fertilizante = st.selectbox("Uso de Fertilizante", opciones_fertilizante)
    riego = st.selectbox("Tipo de Riego", opciones_riego)
    plagas = st.selectbox("Nivel de Plagas", opciones_plagas)
    duracion = st.slider("Duración del cultivo (días)", 110, 150, 130)

with col2:
    altitud = st.number_input("Altitud (msnm)", min_value=2000, max_value=4000, value=3000)
    temperatura = st.slider("Temperatura media (°C)", 5.0, 30.0, 16.0)
    precipitacion = st.slider("Precipitación (mm)", 100, 700, 500)
    ph = st.slider("pH del Suelo", 4.5, 8.5, 6.5)
    materia = st.slider("Materia Orgánica (%)", 0.5, 5.0, 3.0)
    dosis = st.slider("Dosis Fertilizante (kg/ha)", 0, 300, 180)

# Botón de predicción
if st.button("🔍 Predecir rendimiento"):
    # Codificar entradas categóricas
    def codificar(valor, opciones):
        return opciones.index(valor)

    entrada = pd.DataFrame([[
        codificar(variedad, opciones_variedad),
        codificar(textura, opciones_textura),
        codificar(fertilizante, opciones_fertilizante),
        codificar(riego, opciones_riego),
        codificar(plagas, opciones_plagas),
        duracion, altitud, temperatura,
        precipitacion, ph, materia, dosis
    ]], columns=[
        'Variedad', 'Textura_Suelo', 'Uso_Fertilizante', 'Riego', 'Plagas',
        'Duración_Días', 'Altitud_msnm', 'Temperatura_Media_C',
        'Precipitación_mm', 'pH_Suelo', 'Materia_Orgánica_%',
        'Dosis_Fertilizante_kg_ha'
    ])

    # Escalar datos
    scaler = StandardScaler()
    entrada_scaled = scaler.fit_transform(entrada)

    # Predicción con ambos modelos
    pred_xgb = modelo_xgb.predict(entrada_scaled)[0]
    pred_lgb = modelo_lgb.predict(entrada_scaled)[0]

    # Mostrar resultados
    st.success(f"📈 Predicción XGBoost: **{pred_xgb:.2f} t/ha**")
    st.success(f"📈 Predicción LightGBM: **{pred_lgb:.2f} t/ha**")
