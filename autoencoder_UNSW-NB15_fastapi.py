from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

# Define el esquema de entrada con los nuevos campos
class FlowData(BaseModel):
    sbytes: float
    sttl: int
    proto: int
    ct_dst_ltm: int
    is_sm_ips_ports: int
    Dintpkt: float
    ct_src_dport_ltm: int
    Dload: float
    Spkts: int
    Dpkts: int
    Smeansz: float

# Cargar modelo y scaler nuevos
autoencoder = load_model("autoencoder_model_newfields.keras")
scaler = joblib.load("autoencoder_scaler_newfields.pkl")

THRESHOLD = 0.01  # Ajusta según validación

@app.post("/predict")
async def predict(flow: FlowData, request: Request):
    body = await request.json()

    features = pd.DataFrame([{
        "sbytes": flow.sbytes,
        "sttl": flow.sttl,
        "proto": flow.proto,
        "ct_dst_ltm": flow.ct_dst_ltm,
        "is_sm_ips_ports": flow.is_sm_ips_ports,
        "Dintpkt": flow.Dintpkt,
        "ct_src_dport_ltm": flow.ct_src_dport_ltm,
        "Dload": flow.Dload,
        "Spkts": flow.Spkts,
        "Dpkts": flow.Dpkts,
        "Smeansz": flow.Smeansz
    }])

    scaled_features = scaler.transform(features)
    reconstructed = autoencoder.predict(scaled_features, verbose=0)
    mse = np.mean(np.square(scaled_features - reconstructed))

    result = "attack" if mse > THRESHOLD else "normal"

    return {
        "prediction": result,
        "reconstruction_error": float(mse)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
