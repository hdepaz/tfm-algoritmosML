from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

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

# Cargar modelo River preentrenado
try:
    with open("river_hst_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print("Error cargando el modelo:", e)
    model = None

@app.post("/predict")
async def predict(flow: FlowData, request: Request):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    body = await request.json()
#    print("Received JSON:", body)

    # Convertimos a diccionario para usar en River
    features = {
        "Dur": float(flow.Dur),
        "Proto": int(flow.Proto),
        "SrcAddr": int(flow.SrcAddr),
        "Sport": int(flow.Sport),
        "DstAddr": int(flow.DstAddr),
        "Dport": int(flow.Dport),
        "TotPkts": int(flow.TotPkts),
        "TotBytes": int(flow.TotBytes),
        "SrcBytes": int(flow.SrcBytes)
    }

#    print("Antes del score:", features)

    # Obtener score antes de entrenar
    score = model.score_one(features)
#    print("Score:", score)

    # Entrenar modelo en línea
    model.learn_one(features)

    # Clasificación basada en umbral
    threshold = 0.7
    result = "attack" if score > threshold else "normal"

#    print("Después del aprendizaje:", result)

    return {"score": score, "prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
