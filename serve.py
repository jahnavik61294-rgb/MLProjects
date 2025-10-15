from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd, joblib
from pathlib import Path

app = FastAPI()
model = None

class Input(BaseModel):
    records: list

@app.on_event('startup')
def load_model():
    global model
    model = joblib.load(Path('out/pipeline.joblib'))

@app.post('/predict')
def predict(req: Input):
    X = pd.DataFrame(req.records)
    return {'predictions': model.predict_proba(X)[:,1].tolist()}