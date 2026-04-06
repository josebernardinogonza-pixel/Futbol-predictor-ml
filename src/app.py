import streamlit as st
import joblib
import pandas as pd

st.title("⚽ Predictor Inteligente Global")

liga_seleccionada = st.selectbox("Selecciona la Liga", ["Premier League", "La Liga", "Serie A", "Bundesliga", "Liga MX"])

# El resto del código de predicción se mantiene igual, 
# pero ahora el usuario sabe que el modelo fue entrenado con datos de todo el mundo.
st.title("⚽ Predictor de Fútbol con IA")
st.write("Este modelo predice el resultado basado en el promedio de goles de los últimos 3 partidos.")

# 1. Cargar el modelo generado por GitHub Actions
try:
    model = joblib.load('models/soccer_model.pkl')
    
    # 2. Interfaz de usuario
    col1, col2 = st.columns(2)
    
    with col1:
        goles_local = st.number_input("Goles promedio Local", 0.0, 5.0, 1.2)
    with col2:
        goles_visita = st.number_input("Goles promedio Visitante", 0.0, 5.0, 1.0)

    if st.button("Predecir Resultado"):
        # Crear el dato para el modelo
        X = pd.DataFrame([[goles_local, goles_visita]], 
                         columns=['Promedio_Goles_Local', 'Promedio_Goles_Visitante'])
        
        prediccion = model.predict(X)[0]
        
        # Mostrar resultado bonito
        if prediccion == 'H':
            st.success("🏆 Resultado probable: ¡Gana el Local!")
        elif prediccion == 'A':
            st.error("🚩 Resultado probable: ¡Gana el Visitante!")
        else:
            st.warning("🤝 Resultado probable: Empate")

except Exception as e:
    st.error("Aún no se ha generado el modelo. Espera a que GitHub Actions termine.")
