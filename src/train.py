import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Crear carpeta para el modelo
os.makedirs('models', exist_ok=True)

def train():
    # 1. Cargar datos
    url = "https://www.football-data.co.uk/mmz4281/2324/SP1.csv"
    df = pd.read_csv(url)
    
    # 2. Limpieza básica y creación de columnas (Features)
    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].dropna()
    
    # Creamos promedios simples
    df['Promedio_Goles_Local'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.rolling(3, closed='left').mean())
    df['Promedio_Goles_Visitante'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.rolling(3, closed='left').mean())
    
    df = df.dropna()

    # 3. Entrenar el modelo
    X = df[['Promedio_Goles_Local', 'Promedio_Goles_Visitante']]
    y = df['FTR']
    
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)
    
    # 4. Guardar el modelo
    joblib.dump(model, 'models/soccer_model.pkl')
    print("Modelo entrenado y guardado correctamente.")

if __name__ == "__main__":
    train()
