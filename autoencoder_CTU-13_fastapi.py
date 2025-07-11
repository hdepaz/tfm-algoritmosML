from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import sys
import os
#sys.stderr = open(os.devnull, 'w')
from tensorflow.keras.models import load_model

#sys.stderr = open(os.devnull, 'w')
app = FastAPI()

# Define el esquema de entrada
class FlowData(BaseModel):
    Dur: float
    Proto: int
    SrcAddr: int
    Sport: int
    DstAddr: int
    Dport: int
    TotPkts: int
    TotBytes: int
    SrcBytes: int

# Cargar modelo y scaler
autoencoder = load_model("autoencoder_model.keras")
scaler = joblib.load("autoencoder_scaler.pkl")

# Umbral para marcar anomalías (ajústalo según tu dataset y resultados)
THRESHOLD = 0.01

@app.post("/predict")
async def predict(flow: FlowData, request: Request):
    # Extrae los datos del JSON como DataFrame
    body = await request.json()
#    print("Received JSON:", body)

    features = pd.DataFrame([{
        "Dur": flow.Dur,
        "Proto": flow.Proto,
        "SrcAddr": flow.SrcAddr,
        "Sport": flow.Sport,
        "DstAddr": flow.DstAddr,
        "Dport": flow.Dport,
        "TotPkts": flow.TotPkts,
        "TotBytes": flow.TotBytes,
        "SrcBytes": flow.SrcBytes
    }])

    # Normalizar con el mismo scaler del entrenamiento
    scaled_features = scaler.transform(features)

    # Reconstruir con el autoencoder
    reconstructed = autoencoder.predict(scaled_features, verbose=0)

    # Calcular el error de reconstrucción (MSE)
    mse = np.mean(np.square(scaled_features - reconstructed))
#    print(f"MSE: {mse}")

    # Clasificar como ataque si el error es mayor que el umbral
    result = "attack" if mse > THRESHOLD else "normal"

    return {
        "prediction": result,
        "reconstruction_error": mse
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

