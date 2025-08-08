import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Predicción de Rendimiento de Papa", layout="wide")
st.title("🧠 Predicción de Rendimiento de Cultivos de Papa")

# Cargar modelos entrenados
@st.cache_resource
def cargar_modelos():
    modelo_xgb = joblib.load("modelo_xgboost.pkl")
    modelo_lgb = joblib.load("modelo_lightgbm.pkl")
    return modelo_xgb, modelo_lgb

modelo_xgb, modelo_lgb = cargar_modelos()

# Cargar datos
archivo = st.file_uploader("📁 Sube tu archivo CSV con nuevos cultivos", type=["csv"])

if archivo:
    datos = pd.read_csv(archivo)
    st.subheader("📋 Vista previa de los datos cargados")
    st.dataframe(datos.head())

    # Preprocesamiento
    try:
        cat_cols = ['Variedad', 'Textura_Suelo', 'Uso_Fertilizante', 'Riego', 'Plagas']
        for col in cat_cols:
            le = LabelEncoder()
            datos[col] = le.fit_transform(datos[col])

        # Guardar columnas para recuperación
        columnas_originales = datos.columns.tolist()

        # Eliminar columnas no necesarias
        X_nuevos = datos.drop(columns=['ID', 'Fecha_Siembra', 'Fecha_Cosecha'])

        # Escalado
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_nuevos)

        # Predicciones
        y_pred_xgb = modelo_xgb.predict(X_scaled)
        y_pred_lgb = modelo_lgb.predict(X_scaled)

        # Agregar al DataFrame original
        datos['Rendimiento_XGBoost'] = y_pred_xgb
        datos['Rendimiento_LightGBM'] = y_pred_lgb

        # Mostrar resultados
        st.subheader("✅ Predicciones realizadas")
        st.dataframe(datos[['Variedad', 'Rendimiento_XGBoost', 'Rendimiento_LightGBM']].head())

        # Descargar archivo con predicciones
        csv = datos.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar resultados como CSV",
            data=csv,
            file_name='predicciones_rendimiento.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"⚠️ Error en el procesamiento: {e}")
else:
    st.info("Sube un archivo CSV para comenzar la predicción.")
