from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from fastapi import Request
import pandas as pd

app = FastAPI()




class FlowData(BaseModel):
#    StartTime: float    # Puedes cambiarlo a float si usas epoch en float
    Dur: float
    Proto: int
    SrcAddr:int 
    Sport: int
    DstAddr:int 
    Dport: int
    TotPkts: int
    TotBytes: int
    SrcBytes: int



# Carga modelo preentrenado o dummy
try:
    model = joblib.load("isolation_forest_model.pkl")
except:
    model = IsolationForest(contamination=0.1)
    X_dummy = np.random.rand(100, 5)
    model.fit(X_dummy)
    joblib.dump(model, "isolation_forest_model.joblib")

@app.post("/predict")
async def predict(flow: FlowData,request: Request):
    # Extraemos las características numéricas para el modelo
    body = await request.json()
#    print("Received JSON:", body)
#    print (flow)
    features = pd.DataFrame([{
    "Dur": float(flow.Dur),
    "Proto": int(flow.Proto),
    "SrcAddr": int(flow.SrcAddr),     # ya codificada
    "Sport": int(flow.Sport),
    "DstAddr": int(flow.DstAddr),     # ya codificada
    "Dport": int(flow.Dport),
    "TotPkts": int(flow.TotPkts),
    "TotBytes": int(flow.TotBytes),
    "SrcBytes": int(flow.SrcBytes)}])
#    print ("antes",features) 
#    print (features) 
    # Isolation Forest devuelve -1 para anomalías, 1 para normal
    pred = model.predict(features)[0]
#    print (pred) 
    result = "attack" if pred == -1 else "normal"
#    print ("Despues",result)
    return {"prediction": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
