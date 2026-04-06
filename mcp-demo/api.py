# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('iris_model.joblib')
SPECIES = ['setosa', 'versicolor', 'virginica']


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post('/predict')
def predict(data: IrisInput):
    X = np.array([[
        data.sepal_length, data.sepal_width,
        data.petal_length, data.petal_width
    ]])
    prediction = model.predict(X)[0]
    return {'species': SPECIES[prediction]}

@app.post('/predict_proba')
def predict_proba(data: IrisInput):
    X = np.array([[
        data.sepal_length, data.sepal_width,
        data.petal_length, data.petal_width
    ]])
    probs = model.predict_proba(X)[0]
    return {species: round(float(p), 2) for species, p in zip(SPECIES, probs)}
