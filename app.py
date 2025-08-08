import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
from datetime import datetime

# =========================
# Configuración de la app
# =========================
st.set_page_config(page_title="Predicción de Rendimiento de Papa", layout="wide")

# =========================
# Carga de modelos y mapeos
# =========================
@st.cache_resource
def cargar_modelos():
    # XGBoost (joblib) + CatBoost (formato .cbm) + mapeos de categorías (dict)
    xgb_model = joblib.load("modelo_xgboost.pkl")
    cat_model = CatBoostRegressor()
    cat_model.load_model("modelo_catboost.cbm")
    ordenes = joblib.load("ordenes_categorias.pkl")  # dict: {col: [categorias_en_orden]}
    return xgb_model, cat_model, ordenes

xgb_model, cat_model, ordenes_categorias = cargar_modelos()

# =========================
# Navegación
# =========================
pagina = st.sidebar.radio("Navegación", ["📈 Predicción", "📋 Historial"])

# =========================
# Página: Predicción
# =========================
if pagina == "📈 Predicción":
    st.title("🌱 Predicción de Rendimiento de Cultivo de Papa")

    st.subheader("👤 Datos del Usuario")
    nombre = st.text_input("Nombre del Usuario")
    fecha = st.date_input("Fecha del Registro", value=datetime.today())
    ubicacion = st.text_input("Ubicación o Región")

    st.subheader("🧪 Parámetros del Cultivo")

    # Opciones provenientes del entrenamiento (orden garantizado)
    opciones_variedad      = ordenes_categorias['Variedad']
    opciones_textura       = ordenes_categorias['Textura_Suelo']
    opciones_fertilizante  = ordenes_categorias['Uso_Fertilizante']
    opciones_riego         = ordenes_categorias['Riego']
    opciones_plagas        = ordenes_categorias['Plagas']

    col1, col2 = st.columns(2)
    with col1:
        variedad      = st.selectbox("Variedad", opciones_variedad)
        textura       = st.selectbox("Textura del Suelo", opciones_textura)
        fertilizante  = st.selectbox("Uso de Fertilizante", opciones_fertilizante)
        riego         = st.selectbox("Tipo de Riego", opciones_riego)
        plagas        = st.selectbox("Nivel de Plagas", opciones_plagas)
        duracion      = st.slider("Duración del cultivo (días)", 110, 150, 130)

    with col2:
        altitud       = st.number_input("Altitud (msnm)", 2000, 4000, 3000)
        temperatura   = st.slider("Temperatura media (°C)", 5.0, 30.0, 16.0)
        precipitacion = st.slider("Precipitación (mm)", 100, 700, 500)
        ph            = st.slider("pH del Suelo", 4.5, 8.5, 6.5)
        materia       = st.slider("Materia Orgánica (%)", 0.5, 5.0, 3.0)
        dosis         = st.slider("Dosis Fertilizante (kg/ha)", 0, 300, 180)

    if st.button("🔍 Predecir rendimiento"):
        # Orden de columnas como en entrenamiento
        cols = [
            'Variedad','Textura_Suelo','Uso_Fertilizante','Riego','Plagas',
            'Duración_Días','Altitud_msnm','Temperatura_Media_C',
            'Precipitación_mm','pH_Suelo','Materia_Orgánica_%','Dosis_Fertilizante_kg_ha'
        ]

        # -------------------------------
        # 1) DataFrame para CatBoost (strings)
        # -------------------------------
        df_cat = pd.DataFrame([[
            variedad, textura, fertilizante, riego, plagas,
            duracion, altitud, temperatura, precipitacion, ph, materia, dosis
        ]], columns=cols)

        # Predicción CatBoost (recibe strings directamente)
        pred_cat = float(cat_model.predict(df_cat)[0])

        # -------------------------------
        # 2) DataFrame para XGBoost (numérico con mapping ordinal)
        # -------------------------------
        df_xgb_num = df_cat.copy()
        # Mapear cada columna categórica según el orden guardado
        for c, orden in ordenes_categorias.items():
            if c in df_xgb_num.columns:
                mapa = {v: i for i, v in enumerate(orden)}
                df_xgb_num[c] = df_xgb_num[c].map(mapa)

        # Asegurar dtypes numéricos en las columnas continuas
        num_cols = [
            'Duración_Días','Altitud_msnm','Temperatura_Media_C',
            'Precipitación_mm','pH_Suelo','Materia_Orgánica_%','Dosis_Fertilizante_kg_ha'
        ]
        for c in num_cols:
            df_xgb_num[c] = pd.to_numeric(df_xgb_num[c], errors='coerce')

        pred_xgb = float(xgb_model.predict(df_xgb_num.values)[0])

        # Mostrar resultados
        st.success(f"🐱 CatBoost: **{pred_cat:.2f} t/ha**")
        st.success(f"🌳 XGBoost: **{pred_xgb:.2f} t/ha**")

        # Guardar en historial con textos originales (más legible)
        fila_hist = {
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
            "Rendimiento_CatBoost": round(pred_cat, 2),
            "Rendimiento_XGBoost": round(pred_xgb, 2)
        }

        try:
            historial = pd.read_csv("historial_predicciones.csv")
        except FileNotFoundError:
            historial = pd.DataFrame()

        historial = pd.concat([historial, pd.DataFrame([fila_hist])], ignore_index=True)
        historial.to_csv("historial_predicciones.csv", index=False)
        st.success("✅ Registro guardado en historial.")

# =========================
# Página: Historial
# =========================
elif pagina == "📋 Historial":
    st.title("📋 Historial de Predicciones Registradas")
    try:
        historial = pd.read_csv("historial_predicciones.csv")
        st.dataframe(historial, use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ No hay registros aún. Realiza una predicción primero.")
