from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app=FastAPI()
with open("model.pkl","rb") as f:
    model=pickle.load(f)
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
@app.get("/")
def home():
    return {'message': 'ML API is running'}
@app.post("/predict")
def predict(data: InputData):
   features=np.array([[data.sepal_length,data.sepal_width,data.petal_length,data.petal_width]])
   prediction=model.predict(features)
   return {'prediction': int(prediction[0])}
