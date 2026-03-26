# immo_eliza_deployment
# ImmoEliza Price Prediction API

This API predicts real estate prices using a LightGBM model trained on Belgian property data.

## Endpoints

### `GET /`
Returns `{"status": "alive"}` if the server is running.

### `GET /predict`
Returns a description of what the POST endpoint expects.

### `POST /predict`
Receives a JSON with house details and returns a price prediction.

#### Input Format (JSON)
```json
{
  "data": {
    "area": int,
    "property-type": "APARTMENT" | "HOUSE" | "OTHERS",
    "rooms-number": int,
    "zip-code": int,
    "land-area": Optional[int],
    "garden": Optional[bool],
    "garden-area": Optional[int],
    "equipped-kitchen": Optional[bool],
    "full-address": Optional[str],
    "swimming-pool": Optional[bool],
    "furnished": Optional[bool],
    "open-fire": Optional[bool],
    "terrace": Optional[bool],
    "terrace-area": Optional[int],
    "facades-number": Optional[int],
    "building-state": Optional[
      "NEW" | "GOOD" | "TO RENOVATE" | "JUST RENOVATED" | "TO REBUILD"
    ]
  }
}

https://immo-eliza-deployment-qk51.onrender.com
