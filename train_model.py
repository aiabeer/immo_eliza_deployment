# train_model.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/cleaned_dataset.csv')

# Drop Locality column if it exists
if 'Locality' in df.columns:
    df.drop(columns='Locality', inplace=True)

# -----------------------------------------------------------------------------
# Define features (use all columns except target and excluded columns)
# -----------------------------------------------------------------------------
exclude_cols = ['Price', 'State of the property']
base_features = [col for col in df.columns if col not in exclude_cols]

# -----------------------------------------------------------------------------
# Prepare target (Price)
# -----------------------------------------------------------------------------
target = 'Price'
df = df.dropna(subset=[target])   # just in case there are rows without price

X = df[base_features]
y = df[target]

# Log‑transform target to reduce skewness
y_log = np.log(y)

# -----------------------------------------------------------------------------
# Train/test split
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

# -----------------------------------------------------------------------------
# Train XGBoost
# -----------------------------------------------------------------------------
print("Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0   # suppress XGBoost messages
)

# Fit with early stopping
model.fit(X_train, y_train) 

# -----------------------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------------------
y_pred_log = model.predict(X_test)
y_pred = np.exp(y_pred_log)
y_test_orig = np.exp(y_test)

print("Model performance (original scale):")
print(f"MAE: {mean_absolute_error(y_test_orig, y_pred):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_orig, y_pred)):,.2f}")
print(f"R²: {r2_score(y_test_orig, y_pred):.4f}")

# -----------------------------------------------------------------------------
# Save model and feature list
# -----------------------------------------------------------------------------
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/feature_columns.pkl', 'wb') as f:
    pickle.dump(base_features, f)

print("Model saved to model/model.pkl")
print("Feature columns saved to model/feature_columns.pkl")