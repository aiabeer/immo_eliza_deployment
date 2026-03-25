import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
from predict.prediction import predict   # your existing prediction module

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="ImmoEliza Price Predictor")

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# ======================
# HTML form routes
# ======================
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the interactive form at the root."""
    html_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(html_path, "r") as f:
        return f.read()

@app.get("/form", response_class=HTMLResponse)
async def form():
    """Alternative URL to access the form."""
    html_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(html_path, "r") as f:
        return f.read()

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
# API routes
# ======================
@app.get("/alive")
def alive():
    return {"status": "alive"}

@app.get("/predict")
def predict_info():
    return {"message": "POST to /predict with a JSON body containing the house data."}

@app.post("/predict")
def predict_price(data: HouseData):
    input_data = data.dict(by_alias=True)
    try:
        prediction = predict(input_data)
        return {"prediction": prediction, "status_code": 200}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))