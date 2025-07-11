from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pickle

app = FastAPI()

class FlowData(BaseModel):
    sbytes: float
    sttl: float
    proto: int
    ct_dst_ltm: float
    is_sm_ips_ports: float
    Dintpkt: float
    ct_src_dport_ltm: float
    Dload: float
    Spkts: float
    Dpkts: float
    Smeansz: float

# Cargar modelo River preentrenado
try:
    with open("river_hst_model_newfields.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print("Error cargando el modelo:", e)
    model = None

@app.post("/predict")
async def predict(flow: FlowData, request: Request):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    body = await request.json()
    # Construir diccionario para River
    features = {
        "sbytes": float(flow.sbytes),
        "sttl": float(flow.sttl),
        "proto": int(flow.proto),
        "ct_dst_ltm": float(flow.ct_dst_ltm),
        "is_sm_ips_ports": float(flow.is_sm_ips_ports),
        "Dintpkt": float(flow.Dintpkt),
        "ct_src_dport_ltm": float(flow.ct_src_dport_ltm),
        "Dload": float(flow.Dload),
        "Spkts": float(flow.Spkts),
        "Dpkts": float(flow.Dpkts),
        "Smeansz": float(flow.Smeansz)
    }

    score = model.score_one(features)
    model.learn_one(features)

    threshold = 0.7
    result = "attack" if score > threshold else "normal"

    return {"score": score, "prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

