import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analista Pro Fútbol", page_icon="📈")

st.title("📈 Analista Profesional de Resultados")

try:
    model = joblib.load('models/soccer_model.pkl')
    stats = pd.read_csv('models/team_stats.csv')

    col1, col2 = st.columns(2)
    
    with col1:
        local = st.selectbox("Equipo Local", sorted(stats['Equipo'].unique()))
        s_h = stats[stats['Equipo'] == local].iloc[0]
        st.write(f"💪 Ataque: {s_h['Atk_Home']:.2f}")
        st.write(f"🛡️ Defensa: {s_h['Def_Home']:.2f}")

    with col2:
        visita = st.selectbox("Equipo Visitante", sorted(stats['Equipo'].unique()))
        s_v = stats[stats['Equipo'] == visita].iloc[0]
        st.write(f"💪 Ataque: {s_v['Atk_Away']:.2f}")
        st.write(f"🛡️ Defensa: {s_v['Def_Away']:.2f}")

    if st.button("REALIZAR PREDICCIÓN PROFESIONAL"):
        # Preparar entrada para el modelo
        input_data = pd.DataFrame([[s_h['Atk_Home'], s_h['Def_Home'], s_v['Atk_Away'], s_v['Def_Away']]], 
                                 columns=['Atk_Home', 'Def_Home', 'Atk_Away', 'Def_Away'])
        
        probs = model.predict_proba(input_data)[0]
        
        # Mostrar Probabilidades
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Victoria {local}", f"{probs[2]*100:.1f}%")
        c2.metric("Empate", f"{probs[1]*100:.1f}%")
        c3.metric(f"Victoria {visita}", f"{probs[0]*100:.1f}%")

        # --- LÓGICA DE MARCADOR PROBABLE ---
        # Estimamos goles basados en las fuerzas (Simplificación de Poisson)
        goles_l = round((s_h['Atk_Home'] * s_v['Def_Away']) * 1.3, 0)
        goles_v = round((s_v['Atk_Away'] * s_h['Def_Home']) * 1.0, 0)
        
        st.info(f"🏟️ Marcador sugerido por la IA: **{local} {int(goles_l)} - {int(goles_v)} {visita}**")

except Exception as e:
    st.error("Configurando datos del servidor... Refresca en 1 minuto.")
