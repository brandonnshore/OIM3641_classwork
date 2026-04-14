from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
import pandas as pd

# Load the trained pipeline once at startup (not per-request)
model = load_model("best_pipeline")

app = FastAPI(title="Wine Quality Classifier")


class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.get("/")
def root():
    return {"message": "Wine Quality API. POST to /predict or visit /docs"}


@app.post("/predict")
def predict(wine: WineFeatures):
    # PyCaret expects column names with spaces (matches training data)
    input_df = pd.DataFrame([{
        "fixed acidity": wine.fixed_acidity,
        "volatile acidity": wine.volatile_acidity,
        "citric acid": wine.citric_acid,
        "residual sugar": wine.residual_sugar,
        "chlorides": wine.chlorides,
        "free sulfur dioxide": wine.free_sulfur_dioxide,
        "total sulfur dioxide": wine.total_sulfur_dioxide,
        "density": wine.density,
        "pH": wine.pH,
        "sulphates": wine.sulphates,
        "alcohol": wine.alcohol,
    }])

    result = predict_model(model, data=input_df)
    prediction = int(result["prediction_label"].iloc[0])
    confidence = float(result["prediction_score"].iloc[0])

    return {
        "prediction": prediction,
        "label": "good" if prediction == 1 else "not good",
        "confidence": round(confidence, 4),
    }