import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs('models', exist_ok=True)

def train():
    # Diccionario de ligas (Inglaterra, España, Italia, Alemania)
    # Nota: Liga MX no está en este servidor gratuito, pero te diré cómo añadirla abajo.
    urls = {
        "Premier": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "LaLiga": "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
        "SerieA": "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
        "Bundesliga": "https://www.football-data.co.uk/mmz4281/2324/D1.csv"
    }
    
    all_data = []

    for liga, url in urls.items():
        print(f"Descargando datos de {liga}...")
        df_temp = pd.read_csv(url)
        df_temp['Liga'] = liga # Etiquetamos la liga
        all_data.append(df_temp)

    # Unimos todas las ligas en un solo DataFrame gigante
    df = pd.concat(all_data, ignore_index=True)
    
    # Limpieza
    cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Liga']
    df = df[cols].dropna()
    
    # Ingeniería de características (Forma actual)
    df['Promedio_Goles_Local'] = df.groupby(['Liga', 'HomeTeam'])['FTHG'].transform(lambda x: x.rolling(3, closed='left').mean())
    df['Promedio_Goles_Visitante'] = df.groupby(['Liga', 'AwayTeam'])['FTAG'].transform(lambda x: x.rolling(3, closed='left').mean())
    
    df = df.dropna()

    # Entrenar el modelo con datos internacionales
    X = df[['Promedio_Goles_Local', 'Promedio_Goles_Visitante']]
    y = df['FTR']
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    joblib.dump(model, 'models/soccer_model.pkl')
    print("¡Sistema Internacional entrenado!")

if __name__ == "__main__":
    train()
