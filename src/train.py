import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

os.makedirs('models', exist_ok=True)

def train():
    urls = {
        "Premier": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "LaLiga": "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
        "SerieA": "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
        "Bundesliga": "https://www.football-data.co.uk/mmz4281/2324/D1.csv"
    }
    
    all_data = []
    for liga, url in urls.items():
        try:
            df_l = pd.read_csv(url)
            df_l['Liga'] = liga
            all_data.append(df_l)
        except: continue

    df = pd.concat(all_data, ignore_index=True)
    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Liga']].dropna()

    # --- CÃÂLCULO DE FUERZA DE ATAQUE Y DEFENSA (MÃÂTODO PROFESIONAL) ---
    # Promedio de goles de la liga
    avg_home_g = df['FTHG'].mean()
    avg_away_g = df['FTAG'].mean()

    # Fuerza de ataque local: (Goles hechos en casa / Promedio liga en casa)
    home_attack = df.groupby('HomeTeam')['FTHG'].mean() / avg_home_g
    # Fuerza de defensa local: (Goles recibidos en casa / Promedio liga fuera)
    home_defense = df.groupby('HomeTeam')['FTAG'].mean() / avg_away_g
    
    # Fuerza de ataque visitante: (Goles hechos fuera / Promedio liga fuera)
    away_attack = df.groupby('AwayTeam')['FTAG'].mean() / avg_away_g
    # Fuerza de defensa visitante: (Goles recibidos fuera / Promedio liga en casa)
    away_defense = df.groupby('AwayTeam')['FTHG'].mean() / avg_home_g

    # Guardar estas mÃÂ©tricas para la App
    stats = pd.DataFrame({
        'Atk_Home': home_attack, 'Def_Home': home_defense,
        'Atk_Away': away_attack, 'Def_Away': away_defense
    }).fillna(1.0).reset_index()
    stats.columns = ['Equipo', 'Atk_Home', 'Def_Home', 'Atk_Away', 'Def_Away']
    stats.to_csv('models/team_stats.csv', index=False)

    # --- ENTRENAMIENTO DEL MODELO ---
    # Creamos las variables de entrenamiento basadas en fuerzas
    train_df = df.copy()
    train_df = train_df.merge(stats[['Equipo', 'Atk_Home', 'Def_Home']], left_on='HomeTeam', right_on='Equipo')
    train_df = train_df.merge(stats[['Equipo', 'Atk_Away', 'Def_Away']], left_on='AwayTeam', right_on='Equipo')

    X = train_df[['Atk_Home', 'Def_Home', 'Atk_Away', 'Def_Away']]
    y = train_df['FTR'] # H, D, A

    # Usamos Gradient Boosting (mÃÂ¡s potente que Random Forest para esto)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    
    joblib.dump(model, 'models/soccer_model.pkl')
    print("Ã¢ÂÂ Sistema de Alta PrecisiÃÂ³n entrenado.")

if __name__ == "__main__":
    train()
