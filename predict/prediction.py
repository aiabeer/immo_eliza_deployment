import pickle
import numpy as np
from preprocessing.cleaning_data import preprocess

# Load model once at startup
with open('model/lgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(raw_data):
    """
    raw_data: dict from API input.
    Returns predicted price (float).
    """
    X = preprocess(raw_data)
    log_pred = model.predict(X)[0]
    return np.exp(log_pred)  # convert from log back to original scale