import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analista Pro FÃÂºtbol", page_icon="Ã°ÂÂÂ")

st.title("Ã°ÂÂÂ Analista Profesional de Resultados")

try:
    model = joblib.load('models/soccer_model.pkl')
    stats = pd.read_csv('models/team_stats.csv')

    col1, col2 = st.columns(2)
    
    with col1:
        local = st.selectbox("Equipo Local", sorted(stats['Equipo'].unique()))
        s_h = stats[stats['Equipo'] == local].iloc[0]
        st.write(f"Ã°ÂÂÂª Ataque: {s_h['Atk_Home']:.2f}")
        st.write(f"Ã°ÂÂÂ¡Ã¯Â¸Â Defensa: {s_h['Def_Home']:.2f}")

    with col2:
        visita = st.selectbox("Equipo Visitante", sorted(stats['Equipo'].unique()))
        s_v = stats[stats['Equipo'] == visita].iloc[0]
        st.write(f"Ã°ÂÂÂª Ataque: {s_v['Atk_Away']:.2f}")
        st.write(f"Ã°ÂÂÂ¡Ã¯Â¸Â Defensa: {s_v['Def_Away']:.2f}")

    if st.button("REALIZAR PREDICCIÃÂN PROFESIONAL"):
        # Preparar entrada para el modelo
        input_data = pd.DataFrame([[s_h['Atk_Home'], s_h['Def_Home'], s_v['Atk_Away'], s_v['Def_Away']]], 
                                 columns=['Atk_Home', 'Def_Home', 'Atk_Away', 'Def_Away'])
        
        # Obtener probabilidades: [A (Visitante), D (Empate), H (Local)]
        probs = model.predict_proba(input_data)[0]
        
        # Mostrar Probabilidades CON LOS NOMBRES CORRECTOS
        st.markdown("---")
        st.subheader(f"AnÃÂ¡lisis detallado: {local} vs {visita}")
        
        c1, c2, c3 = st.columns(3)
        # probs[2] es Local, probs[1] es Empate, probs[0] es Visitante
        c1.metric(f"Victoria {local}", f"{probs[2]*100:.1f}%")
        c2.metric("Empate", f"{probs[1]*100:.1f}%")
        c3.metric(f"Victoria {visita}", f"{probs[0]*100:.1f}%")

        # --- LÃÂGICA DE MARCADOR ---
        # Estimamos goles basados en las fuerzas
        goles_l = int(round((s_h['Atk_Home'] * s_v['Def_Away']) * 1.3, 0))
        goles_v = int(round((s_v['Atk_Away'] * s_h['Def_Home']) * 1.0, 0))
        
        st.success(f"Ã°ÂÂÂÃ¯Â¸Â Marcador sugerido por la IA: **{local} {goles_l} - {goles_v} {visita}**")
