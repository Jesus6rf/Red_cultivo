import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from datetime import datetime

# Configuración de página
st.set_page_config(page_title="Predicción de Papa", layout="wide")

# -----------------------
# Cargar modelos y configs
# -----------------------
@st.cache_resource
def cargar_modelos():
    modelo_xgb = joblib.load("modelo_xgboost.pkl")
    modelo_cat = CatBoostRegressor()
    modelo_cat.load_model("modelo_catboost.cbm")
    ordenes_categorias = joblib.load("ordenes_categorias.pkl")
    return modelo_xgb, modelo_cat, ordenes_categorias

modelo_xgb, modelo_cat, ordenes_categorias = cargar_modelos()

# -----------------------
# Pestañas
# -----------------------
pagina = st.sidebar.radio("Navegación", ["📈 Predicción", "📋 Historial"])

# -----------------------
# Página 1: Predicción
# -----------------------
if pagina == "📈 Predicción":
    st.title("🌱 Predicción de Rendimiento de Cultivo de Papa")

    st.subheader("👤 Datos del Usuario")
    nombre = st.text_input("Nombre del Usuario")
    fecha = st.date_input("Fecha del Registro", value=datetime.today())
    ubicacion = st.text_input("Ubicación o Región")

    st.subheader("🧪 Parámetros del Cultivo")

    col1, col2 = st.columns(2)
    with col1:
        variedad = st.selectbox("Variedad", ordenes_categorias['Variedad'])
        textura = st.selectbox("Textura del Suelo", ordenes_categorias['Textura_Suelo'])
        fertilizante = st.selectbox("Uso de Fertilizante", ordenes_categorias['Uso_Fertilizante'])
        riego = st.selectbox("Tipo de Riego", ordenes_categorias['Riego'])
        plagas = st.selectbox("Nivel de Plagas", ordenes_categorias['Plagas'])
        duracion = st.slider("Duración del cultivo (días)", 110, 150, 130)

    with col2:
        altitud = st.number_input("Altitud (msnm)", 2000, 4000, 3000)
        temperatura = st.slider("Temperatura media (°C)", 5.0, 30.0, 16.0)
        precipitacion = st.slider("Precipitación (mm)", 100, 700, 500)
        ph = st.slider("pH del Suelo", 4.5, 8.5, 6.5)
        materia = st.slider("Materia Orgánica (%)", 0.5, 5.0, 3.0)
        dosis = st.slider("Dosis Fertilizante (kg/ha)", 0, 300, 180)

    if st.button("🔍 Predecir rendimiento"):
        # Crear DataFrame igual que en entrenamiento
        entrada_df = pd.DataFrame([{
            'Variedad': variedad,
            'Textura_Suelo': textura,
            'Uso_Fertilizante': fertilizante,
            'Riego': riego,
            'Plagas': plagas,
            'Duración_Días': duracion,
            'Altitud_msnm': altitud,
            'Temperatura_Media_C': temperatura,
            'Precipitación_mm': precipitacion,
            'pH_Suelo': ph,
            'Materia_Orgánica_%': materia,
            'Dosis_Fertilizante_kg_ha': dosis
        }])

        # Mapear variables categóricas a números según orden guardado
        for col in ordenes_categorias:
            entrada_df[col] = entrada_df[col].astype(pd.CategoricalDtype(categories=ordenes_categorias[col]))
            entrada_df[col] = entrada_df[col].cat.codes

        # Predicciones
        pred_xgb = modelo_xgb.predict(entrada_df)[0]
        pred_cat = modelo_cat.predict(entrada_df)[0]

        st.success(f"📈 Predicción XGBoost: **{pred_xgb:.2f} t/ha**")
        st.success(f"📈 Predicción CatBoost: **{pred_cat:.2f} t/ha**")

        # Guardar en historial
        fila = entrada_df.copy()
        fila["Nombre"] = nombre
        fila["Fecha"] = fecha
        fila["Ubicación"] = ubicacion
        fila["Rendimiento_XGBoost"] = round(pred_xgb, 2)
        fila["Rendimiento_CatBoost"] = round(pred_cat, 2)

        try:
            historial = pd.read_csv("historial_predicciones.csv")
        except FileNotFoundError:
            historial = pd.DataFrame()

        historial = pd.concat([historial, fila], ignore_index=True)
        historial.to_csv("historial_predicciones.csv", index=False)
        st.success("✅ Registro guardado en historial.")

# -----------------------
# Página 2: Historial
# -----------------------
elif pagina == "📋 Historial":
    st.title("📋 Historial de Predicciones Registradas")
    try:
        historial = pd.read_csv("historial_predicciones.csv")
        st.dataframe(historial)
    except FileNotFoundError:
        st.warning("⚠️ No hay registros aún. Realiza una predicción primero.")
