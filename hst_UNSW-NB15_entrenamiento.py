import pandas as pd
from river import anomaly
from river import preprocessing
import pickle

# Cargar datos
print("Cargando datos...")
df = pd.read_csv("/home/hdepaz/hdepaz/master/tfm2/UNSW-NB15/CSV-Files/UNSW-datos_entrenamiento.csv")


df = df.fillna(0)
print("Nmero de filas:", df.shape[0])
df = df[df["Label"] != "attack"]
print("Nmero de filas:", df.shape[0])
df = df.drop(columns=["Label"])



# Convertir IPs a enteros (si es que tienes SrcAddr y DstAddr con IP string)
def ip_to_int(ip):
    try:
        return int.from_bytes(map(int, ip.split('.')), byteorder='big')
    except:
        return 0

df = df.fillna(0)

feature_columns = [
    "sbytes", "sttl", "proto", "ct_dst_ltm", "is_sm_ips_ports",
    "Dintpkt", "ct_src_dport_ltm", "Dload", "Spkts", "Dpkts",
    "Smeansz"
]

# Pero quitamos la etiqueta para entrenar:

print("Creando dataset...")
dataset = df[feature_columns].to_dict(orient="records")

print("Escalando y entrenando HalfSpaceTrees...")
pipeline = preprocessing.MinMaxScaler() | anomaly.HalfSpaceTrees(
    n_trees=40,
    height=10,
    window_size=250,
    seed=42
)

for i, x in enumerate(dataset):
    if i % 1000 == 0:
        print(f"Procesando {i} de {len(dataset)}")
    pipeline.learn_one(x)

# Guardar modelo
with open("river_hst_model_newfields.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Modelo entrenado y guardado.")

