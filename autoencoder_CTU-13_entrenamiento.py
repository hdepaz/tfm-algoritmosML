import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
import joblib

# Leer CSV
df = pd.read_csv("/home/hdepaz/hdepaz/master/tfm2/fuentes/entrenamiento-CTU13/input_entrenamiento.csv")

# Preparamos el dataset para el entrenamiento 
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

# Rellenar NaNs si existen
df = df.fillna(0)

# Features
feature_columns = [
    "Dur", "Proto", "SrcAddr", "Sport",
    "DstAddr", "Dport", "TotPkts", "TotBytes", "SrcBytes"
]

X = df[feature_columns].astype(np.float32)

# Escalar los datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Guardar el scaler
joblib.dump(scaler, "autoencoder_scaler.pkl")

# Arquitectura del Autoencoder
input_dim = X_scaled.shape[1]
encoding_dim = 5  

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(1e-5))(input_layer)
decoded = Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Entrenar el modelo
autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# Guardar el modelo
autoencoder.save("autoencoder_model.keras")
