from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

app = FastAPI()

# Clase con los nuevos campos para predicción
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

# Cargar modelo entrenado
try:
    model = joblib.load("isolation_forest_model.pkl")
except:
    model = IsolationForest(contamination=0.02)
    X_dummy = np.random.rand(100, 11)
    model.fit(X_dummy)
    joblib.dump(model, "isolation_forest_model.pkl")

@app.post("/predict")
async def predict(flow: FlowData, request: Request):
    # Convertir input a DataFrame con los campos correctos
    features = pd.DataFrame([{
        "sbytes": float(flow.sbytes),
        "sttl": int(flow.sttl),
        "proto": int(flow.proto),
        "ct_dst_ltm": int(flow.ct_dst_ltm),
        "is_sm_ips_ports": int(flow.is_sm_ips_ports),
        "Dintpkt": float(flow.Dintpkt),
        "ct_src_dport_ltm": int(flow.ct_src_dport_ltm),
        "Dload": float(flow.Dload),
        "Spkts": int(flow.Spkts),
        "Dpkts": int(flow.Dpkts),
        "Smeansz": float(flow.Smeansz)
    }])

    # Predicción
    pred = model.predict(features)[0]
    result = "attack" if pred == -1 else "normal"
    return {"prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
