import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Leer CSV ya preprocesado
df = pd.read_csv("/home/hdepaz/hdepaz/master/tfm2/UNSW-NB15/CSV-Files/UNSW-datos_entrenamiento.csv")

# Eliminar la columna Label (no se usa para entrenamiento)
df = df.drop(columns=["Label"])

# Rellenar NaNs si existen
df = df.fillna(0)

# Definir nuevas columnas de características (features)
feature_columns = [
    "sbytes", "sttl", "proto", "ct_dst_ltm", "is_sm_ips_ports",
    "Dintpkt", "ct_src_dport_ltm", "Dload", "Spkts", "Dpkts",
    "Smeansz"
]

# Preparar los datos para entrenamiento
X = df[feature_columns]

# Entrenar Isolation Forest
model = IsolationForest(
    n_estimators=600,
    max_samples=100000,
    contamination=0.07,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X)

# Guardar modelo entrenado
joblib.dump(model, "isolation_forest_model.pkl")

print("✅ Modelo Isolation Forest entrenado y guardado.")
