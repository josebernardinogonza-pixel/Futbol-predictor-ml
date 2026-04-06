import streamlit as st
import joblib
import pandas as pd

# 1. Configuración de la página (Título en la pestaña del navegador)
st.set_page_config(page_title="Football Predictor Pro", page_icon="⚽", layout="centered")

# 2. Estilo CSS para que se vea "Pro" (Colores y bordes)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #2e7bcf;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #1e2130;
        border: 1px solid #3e445e;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Encabezado con Logo/Emoji
st.title("⚽ Predictor Inteligente Global")
st.markdown("---")

# 4. Cargar el modelo
try:
    model = joblib.load('models/soccer_model.pkl')
    
    # --- BARRA LATERAL (Sidebar) ---
    st.sidebar.header("Configuración")
    liga = st.sidebar.selectbox("Selecciona la Liga", 
                                ["Premier League", "La Liga", "Serie A", "Bundesliga", "Liga MX"])
    st.sidebar.info("Este modelo analiza las 4 ligas top de Europa y la Liga MX en tiempo real.")

    # --- CUERPO PRINCIPAL ---
    st.subheader(f"Análisis de Partido: {liga}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏠 Equipo Local")
        goles_h = st.slider("Goles promedio (últ. 3 partidos)", 0.0, 5.0, 1.5, step=0.1, key="h")
        st.caption("Eficacia ofensiva en casa")

    with col2:
        st.markdown("### 🚩 Equipo Visitante")
        goles_v = st.slider("Goles promedio (últ. 3 partidos)", 0.0, 5.0, 1.0, step=0.1, key="v")
        st.caption("Eficacia ofensiva fuera")

    st.markdown("---")

    # Botón de Predicción
    if st.button("CALCULAR PROBABILIDADES"):
        # Preparar datos para el modelo
        X = pd.DataFrame([[goles_h, goles_v]], 
                         columns=['Promedio_Goles_Local', 'Promedio_Goles_Visitante'])
        
        # Obtener probabilidades (Predict Proba)
        # El orden suele ser: [Gana Visitante (A), Empate (D), Gana Local (H)]
        probs = model.predict_proba(X)[0]
        
        # Mostrar resultados con barras de progreso
        st.markdown("### 📊 Pronóstico del Modelo")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Victoria Local", f"{probs[2]*100:.1f}%")
            st.progress(probs[2])
        with c2:
            st.metric("Empate", f"{probs[1]*100:.1f}%")
            st.progress(probs[1])
        with c3:
            st.metric("Victoria Visita", f"{probs[0]*100:.1f}%")
            st.progress(probs[0])

        # Mensaje Final según la probabilidad más alta
        max_prob = max(probs)
        if probs[2] == max_prob:
            st.success(f"✅ El análisis sugiere una alta probabilidad de **Victoria para el Local** en la {liga}.")
        elif probs[0] == max_prob:
            st.error(f"🚀 El análisis sugiere una alta probabilidad de **Victoria para el Visitante** en la {liga}.")
        else:
            st.warning(f"🤝 El análisis sugiere un resultado muy ajustado: **Empate** en la {liga}.")

except Exception as e:
    st.error("⚠️ El modelo de IA se está configurando. Por favor, espera a que GitHub Actions termine su primera ejecución.")
    st.info("Esto puede tardar un par de minutos después del último commit.")

# 5. Pie de página
st.markdown("---")
st.caption("Desarrollado con Python, Scikit-Learn y GitHub Actions (MLOps). Datos actualizados semanalmente.")
