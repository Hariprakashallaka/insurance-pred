from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# 1. Load the trained pipeline
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "insurance_model.pkl"

model = joblib.load(MODEL_PATH)


# 2. Create FastAPI application
app = FastAPI(
    title="Medical Insurance Charges Prediction API",
    description="Predict insurance charges using Linear Regression.",
    version="1.0.0",
)


# 3. Define the input format
class InsuranceData(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: Literal["male", "female"]
    bmi: float = Field(gt=0, le=100)
    children: int = Field(ge=0, le=20)
    smoker: Literal["yes", "no"]
    region: Literal[
        "southwest",
        "southeast",
        "northwest",
        "northeast",
    ]


# 4. Home endpoint
@app.get("/")
def home():
    return {
        "message": "Insurance Charges Prediction API is running",
        "docs": "/docs",
    }


# 5. Prediction endpoint
@app.post("/predict")
def predict_charges(data: InsuranceData):

    # Convert validated input into a DataFrame
    input_data = pd.DataFrame([data.model_dump()])

    # Predict insurance charges
    prediction = model.predict(input_data)[0]

    return {
        "predicted_charges": round(float(prediction), 2)
    }
