import numpy as np
import pickle
import os

# Load feature columns, income map, and median income
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
with open(os.path.join(MODEL_DIR, 'feature_columns.pkl'), 'rb') as f:
    FEATURE_COLUMNS = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'income_map.pkl'), 'rb') as f:
    INCOME_MAP = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'median_income.pkl'), 'rb') as f:
    MEDIAN_INCOME = pickle.load(f)

# Mapping for building-state string to numeric code
STATE_MAPPING = {
    "NEW": 1,
    "GOOD": 2,
    "TO RENOVATE": 3,
    "JUST RENOVATED": 4,
    "TO REBUILD": 5
}

def preprocess(raw_data):
    """
    Convert raw API input into a feature vector expected by the model.
    raw_data: dict with keys matching API input fields.
    Returns a 2D numpy array (1 row) of features.
    """
    # Extract fields
    area = raw_data.get('area', 0)
    property_type = raw_data.get('property-type', 'OTHERS')
    rooms_number = raw_data.get('rooms-number', 0)
    zip_code = raw_data.get('zip-code', 0)
    land_area = raw_data.get('land-area', 0)
    garden = bool(raw_data.get('garden', False))
    garden_area = raw_data.get('garden-area', 0) if garden else 0
    equipped_kitchen = bool(raw_data.get('equipped-kitchen', False))
    swimming_pool = bool(raw_data.get('swimming-pool', False))
    furnished = bool(raw_data.get('furnished', False))
    open_fire = bool(raw_data.get('open-fire', False))
    terrace = bool(raw_data.get('terrace', False))
    terrace_area = raw_data.get('terrace-area', 0) if terrace else 0
    facades_number = raw_data.get('facades-number', 0)
    building_state_str = raw_data.get('building-state', 'NEW')
    building_state = STATE_MAPPING.get(building_state_str, 1)

    # Get income from zip code
    municipality_income = INCOME_MAP.get(zip_code, MEDIAN_INCOME)

    # Create binary flags for property type
    type_house = 1 if property_type.upper() == 'HOUSE' else 0
    type_apartment = 1 if property_type.upper() == 'APARTMENT' else 0

    # Build feature dictionary with the exact names used during training
    features = {
        'Livable surface': area,
        'Number of rooms': rooms_number,
        'municipality_income': municipality_income,
        'Garden area': garden_area,
        'Garden': int(garden),
        'Terrace': int(terrace),
        'Surface terrace': terrace_area,
        'Fully equipped kitchen': int(equipped_kitchen),
        'Swimming pool': int(swimming_pool),
        'Furnished': int(furnished),
        'Fireplace': int(open_fire),
        'Number of facades': facades_number,
        'State of the property': building_state,
        'type_house': type_house,
        'type_apartement': type_apartment
    }

    # Add interaction features (must match training)
    features['liv_surf_facades'] = area * facades_number
    features['liv_surf_rooms'] = area * rooms_number
    features['garden_area_flag'] = garden_area * int(garden)
    features['terrace_surface_flag'] = terrace_area * int(terrace)

    # Add squared terms
    features['Livable surface_squared'] = area ** 2
    features['Number of rooms_squared'] = rooms_number ** 2
    features['Garden area_squared'] = garden_area ** 2

    # Ensure all expected features are present, fill missing with 0
    feature_vector = [features.get(col, 0) for col in FEATURE_COLUMNS]

    return np.array(feature_vector).reshape(1, -1)