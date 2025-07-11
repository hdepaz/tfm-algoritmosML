
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Leer CSV
df = pd.read_csv("/home/hdepaz/hdepaz/master/tfm2/fuentes/entrenamiento-CTU13/input_entrenamiento.csv")

#Preparamos el dataframe para el entrenamiento, sin horas de inicio, sin trafico malicioso y sin la columna de etiquetas
df = df.drop(columns=["StartTime"])
#df = df[df["Label"] != "Botnet"]
df = df.drop(columns=["Label"])


# Convertir IPs a enteros
def ip_to_int(ip):
    try:
        return int.from_bytes(map(int, ip.split('.')), byteorder='big')
    except:
        return 0  # fallback

df["SrcAddr"] = df["SrcAddr"].apply(ip_to_int)
df["DstAddr"] = df["DstAddr"].apply(ip_to_int)

# Rellenar NaNs si existen
df = df.fillna(0)

# Definir features
feature_columns = [
    "Dur", "Proto", "SrcAddr", "Sport",
    "DstAddr", "Dport", "TotPkts", "TotBytes", "SrcBytes"
]

X = df[feature_columns]

# Entrenar Isolation Forest
#model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model = IsolationForest(n_estimators=600, max_samples=100000, contamination=0.02, max_features=0.8, random_state=42,n_jobs=-1 ) 
model.fit(X)

# Guardar modelo
joblib.dump(model, "isolation_forest_model.pkl")
                                                        
