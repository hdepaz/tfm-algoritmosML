import pandas as pd
from river import anomaly
from river import preprocessing
import pickle

# Cargar datos
print ("Cargando datos...")
df = pd.read_csv("/home/hdepaz/hdepaz/master/tfm2/fuentes/entrenamiento-CTU13/input_entrenamiento.csv")
total=len(df)

print ("Filtrando datos...")
# Eliminar StartTime si existe
df = df.drop(columns=["StartTime"])
df = df[df["Label"] != "Botnet"]
df = df.drop(columns=["Label"])


# Convertir IPs a enteros
def ip_to_int(ip):
    try:
        return int.from_bytes(map(int, ip.split('.')), byteorder='big')
    except:
        return 0

df["SrcAddr"] = df["SrcAddr"].apply(ip_to_int)
df["DstAddr"] = df["DstAddr"].apply(ip_to_int)

# Rellenar NaNs
df = df.fillna(0)

# Seleccionar columnas
feature_columns = [
    "Dur", "Proto", "SrcAddr", "Sport",
    "DstAddr", "Dport", "TotPkts", "TotBytes", "SrcBytes"
]

print ("Creando dataset...")
# Crear dataset
dataset = df[feature_columns].to_dict(orient="records")

# Crear pipeline: escalador + modelo
print ("Escalando...")
pipeline = preprocessing.MinMaxScaler() | anomaly.HalfSpaceTrees(
    n_trees=40,
    height=8,
    window_size=250,
    seed=42
)
cont=0
for x in dataset:
    print ("Procesando ",cont , " de ", total)
    pipeline.learn_one(x)          # entrenamiento
    cont=cont+1

# Guardar modelo entrenado
with open("river_hst_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Modelo entrenado y guardado.")


