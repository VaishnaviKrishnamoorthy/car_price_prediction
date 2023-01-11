from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
 
app = FastAPI()
 
class datas(BaseModel):
    Year : int
    Present_Price : float
    Kms_Driven  :  int
    Fuel_Type  :  int
    Seller_Type : int
    Transmission : int
    Owner  : int
    

@app.post('/predict')
async def predict(value: datas):
    data = value.dict()
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    data_in = [[data['Year'], data['Present_Price'], data['Kms_Driven'], data['Fuel_Type'], data['Seller_Type'], data['Transmission'], data['Owner']]]
    prediction = loaded_model.predict(data_in)
    output = prediction[0].astype('int')
    return{'you can sell the car for {}'.format(output)}