from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from predict.prediction import predict
import numpy as np

app = FastAPI(title="ImmoEliza Price Predictor")

# ======================
# Input Schema
# ======================
class HouseData(BaseModel):
    area: int
    property_type: str = Field(..., alias="property-type")
    rooms_number: int = Field(..., alias="rooms-number")
    zip_code: int = Field(..., alias="zip-code")
    land_area: Optional[int] = Field(None, alias="land-area")
    garden: Optional[bool] = None
    garden_area: Optional[int] = Field(None, alias="garden-area")
    equipped_kitchen: Optional[bool] = Field(None, alias="equipped-kitchen")
    full_address: Optional[str] = Field(None, alias="full-address")
    swimming_pool: Optional[bool] = Field(None, alias="swimming-pool")
    furnished: Optional[bool] = None
    open_fire: Optional[bool] = Field(None, alias="open-fire")
    terrace: Optional[bool] = None
    terrace_area: Optional[int] = Field(None, alias="terrace-area")
    facades_number: Optional[int] = Field(None, alias="facades-number")
    building_state: Optional[str] = Field(None, alias="building-state")

# ======================
# Routes
# ======================
@app.get("/")
def alive():
    return {"status": "alive"}

@app.get("/predict")
def predict_info():
    return {"message": "POST to /predict with a JSON body containing the house data."}

@app.post("/predict")
def predict_price(data: HouseData):
    # Convert Pydantic model to dict
    input_data = data.dict(by_alias=True)  # using alias to match incoming JSON keys
    try:
        prediction = predict(input_data)
        return {"prediction": prediction, "status_code": 200}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))