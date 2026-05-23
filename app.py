import streamlit as st
import pandas as pd
import pickle
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin

# CONFIGURACIÓN DE PÁGINA

st.set_page_config(
    page_title="Predicción Solar",
    page_icon="☀️",
    layout="wide"
)


# CLASE DateTransformer

class DateTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # Convertir fechas
        X['Data'] = pd.to_datetime(
            X['Data'],
            format='%m/%d/%Y %I:%M:%S %p'
        )

        X['Time'] = pd.to_datetime(
            X['Time'],
            format='%H:%M:%S'
        )

        X['TimeSunRise'] = pd.to_datetime(
            X['TimeSunRise'],
            format='%H:%M:%S'
        )

        X['TimeSunSet'] = pd.to_datetime(
            X['TimeSunSet'],
            format='%H:%M:%S'
        )

        # Extraer componentes fecha
        X['year'] = X['Data'].dt.year
        X['month'] = X['Data'].dt.month
        X['day'] = X['Data'].dt.day

        # Extraer hora
        X['hour'] = X['Time'].dt.hour
        X['minute'] = X['Time'].dt.minute

        # VARIABLES CÍCLICAS

        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)

        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)

        X['day_sin'] = np.sin(2 * np.pi * X['day'] / 31)
        X['day_cos'] = np.cos(2 * np.pi * X['day'] / 31)

        X['minute_sin'] = np.sin(2 * np.pi * X['minute'] / 60)
        X['minute_cos'] = np.cos(2 * np.pi * X['minute'] / 60)

        # VARIABLES AMANECER / ATARDECER

        X['sunrise_hour'] = X['TimeSunRise'].dt.hour
        X['sunrise_minute'] = X['TimeSunRise'].dt.minute

        X['sunset_hour'] = X['TimeSunSet'].dt.hour
        X['sunset_minute'] = X['TimeSunSet'].dt.minute

        # Nuevas variables
        X['daylight_hours'] = (
            X['sunset_hour'] - X['sunrise_hour']
        )

        X['time_since_sunrise'] = (
            X['hour'] - X['sunrise_hour']
        )

        X['time_until_sunset'] = (
            X['sunset_hour'] - X['hour']
        )

        # Eliminar columnas originales
        X = X.drop([
            'UNIXTime',
            'Data',
            'Time',
            'TimeSunRise',
            'TimeSunSet',
            'hour',
            'month',
            'day',
            'minute'
        ], axis=1)

        return X


# CARGAR MODELOS

modelo_rf = joblib.load(
    "modelos/modelo_rf.joblib"
)

modelo_adaboost = pickle.load(
    open("modelos/modelo_adaboost.sav", "rb")
)

modelo_gb = pickle.load(
    open("modelos/modelo_gradient_boosting.sav", "rb")
)

# ENCABEZADO

st.title("☀️ Predicción de Radiación Solar")

st.markdown("""
### UNIVERSIDAD AUTÓNOMA DE CHIHUAHUA  
**Facultad de Ingeniería**  

- **Alumno:** Jasaman Sagarnaga Pérez  
- **Matrícula:** 335977  
- **Asesor:** Olanda Prieto Ordaz  
""")

st.divider()

st.markdown("""
Esta aplicación utiliza modelos de Machine Learning para predecir radiación solar utilizando:

- Random Forest
- AdaBoost
- Gradient Boosting
""")

# SIDEBAR

st.sidebar.header("📥 Ingrese los datos")

date = st.sidebar.text_input(
    "Fecha completa",
    "09/29/2016 12:00:00 PM"
)

time = st.sidebar.text_input(
    "Hora",
    "12:00:00"
)

temperature = st.sidebar.number_input(
    "Temperatura",
    value=30.0
)

pressure = st.sidebar.number_input(
    "Presión",
    value=1013.0
)

humidity = st.sidebar.number_input(
    "Humedad",
    value=50.0
)

wind_direction = st.sidebar.number_input(
    "Dirección del viento",
    value=180.0
)

speed = st.sidebar.number_input(
    "Velocidad del viento",
    value=5.0
)

sunrise = st.sidebar.text_input(
    "Hora amanecer",
    "06:00:00"
)

sunset = st.sidebar.text_input(
    "Hora atardecer",
    "18:00:00"
)

# DATAFRAME

new_data = pd.DataFrame({

    "UNIXTime": [0],

    "Data": [date.strip()],

    "Time": [time.strip()],

    "Temperature": [temperature],

    "Pressure": [pressure],

    "Humidity": [humidity],

    "WindDirection(Degrees)": [wind_direction],

    "Speed": [speed],

    "TimeSunRise": [sunrise.strip()],

    "TimeSunSet": [sunset.strip()]

})

# BOTÓN DE PREDICCIÓN

if st.button("🔍 Predecir Radiación Solar"):

    pred_rf = modelo_rf.predict(new_data)

    pred_adaboost = modelo_adaboost.predict(new_data)

    pred_gb = modelo_gb.predict(new_data)

    # RESULTADOS

    st.subheader("📊 Resultados")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Random Forest",
            value=f"{pred_rf[0]:.2f}"
        )

    with col2:
        st.metric(
            label="AdaBoost",
            value=f"{pred_adaboost[0]:.2f}"
        )

    with col3:
        st.metric(
            label="Gradient Boosting",
            value=f"{pred_gb[0]:.2f}"
        )

    # MEJORES MODELOS

    resultados = {
        "Random Forest": pred_rf[0],
        "AdaBoost": pred_adaboost[0],
        "Gradient Boosting": pred_gb[0]
    }

    mejor_modelo = max(resultados, key=resultados.get)

    st.info(
    "📌 De acuerdo con las métricas obtenidas durante el entrenamiento, el modelo con mejor desempeño general fue Random Forest."
    )

    # DATOS INGRESADOS
    with st.expander("📄 Ver datos ingresados"):

        st.dataframe(new_data)

# Pie de página

st.divider()

st.caption(
    "Proyecto Final de Machine Learning - Predicción de Radiación Solar"
)