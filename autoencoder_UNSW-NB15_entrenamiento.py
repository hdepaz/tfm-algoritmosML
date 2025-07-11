# Entrenamiento autoencoder con nuevos campos
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
import joblib

# Leer CSV
df = pd.read_csv("/home/hdepaz/hdepaz/master/tfm2/UNSW-NB15/CSV-Files/UNSW-datos_entrenamiento.csv")

# No hay "StartTime" ni eliminación de Label si ya no tienes esa columna
# Si la tienes, ajusta según corresponda

# Rellenar NaNs si existen
df = df.fillna(0)
print("Nmero de filas:", df.shape[0])
df = df[df["Label"] != "attack"]
print("Nmero de filas:", df.shape[0])
df = df.drop(columns=["Label"])

# Features nuevos que usas
feature_columns = [
    "sbytes", "sttl", "proto", "ct_dst_ltm", "is_sm_ips_ports",
    "Dintpkt", "ct_src_dport_ltm", "Dload", "Spkts", "Dpkts",
    "Smeansz"
]

X = df[feature_columns].astype(np.float32)

# Escalar datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Guardar scaler
joblib.dump(scaler, "autoencoder_scaler_newfields.pkl")

# Arquitectura autoencoder
input_dim = X_scaled.shape[1]
encoding_dim = 5

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(1e-5))(input_layer)
decoded = Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Entrenar
autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# Guardar modelo
autoencoder.save("autoencoder_model_newfields.keras")
