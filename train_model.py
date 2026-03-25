import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. Load Data
# ======================
df = pd.read_csv('data/sale_with_income.csv')

# ======================
# 2. Select Features Matching API Input
# ======================
keep_cols = [
    'Livable surface', 'Number of rooms', 'municipality_income',  # replaced Locality
    'Garden area', 'Garden', 'Terrace', 'Surface terrace',
    'Fully equipped kitchen', 'Swimming pool', 'Furnished',
    'Fireplace', 'Number of facades', 'State of the property',
    'type_house', 'type_apartement', 'Price'
]
df = df[keep_cols].copy()

# ======================
# 3. Basic Cleaning
# ======================
numeric_cols = ['Livable surface', 'Number of rooms', 'municipality_income',
                'Garden area', 'Surface terrace', 'Number of facades']
bool_cols = ['Garden', 'Terrace', 'Fully equipped kitchen', 'Swimming pool',
             'Furnished', 'Fireplace', 'type_house', 'type_apartement']
categorical_col = ['State of the property']

df[numeric_cols] = df[numeric_cols].fillna(0)
df[bool_cols] = df[bool_cols].fillna(0).astype(int)
df[categorical_col] = df[categorical_col].fillna(1)  # default to NEW

# ======================
# 4. Create Interaction Features
# ======================
df['liv_surf_facades'] = df['Livable surface'] * df['Number of facades']
df['liv_surf_rooms'] = df['Livable surface'] * df['Number of rooms']
df['garden_area_flag'] = df['Garden area'] * df['Garden']
df['terrace_surface_flag'] = df['Surface terrace'] * df['Terrace']

# Add squared terms
for col in ['Livable surface', 'Number of rooms', 'Garden area']:
    df[f'{col}_squared'] = df[col] ** 2

# ======================
# 5. Train/Test Split
# ======================
target = 'Price'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log transform target
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

# ======================
# 6. Hyperparameter Tuning with Optuna
# ======================
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train_log, test_size=0.2, random_state=42
)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }
    model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train_sub, y_train_sub,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, pred))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

# ======================
# 7. Train Final Model
# ======================
best_params = study.best_params
best_params['random_state'] = 42
best_params['n_jobs'] = -1
best_params['verbose'] = -1

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_train, y_train_log,
                eval_set=[(X_test, y_test_log)],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

# ======================
# 8. Evaluate & Save
# ======================
y_pred_log = final_model.predict(X_test)
y_pred = np.exp(y_pred_log)
y_test_orig = np.exp(y_test_log)

print(f"MAE: {mean_absolute_error(y_test_orig, y_pred):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_orig, y_pred)):,.2f}")
print(f"R²: {r2_score(y_test_orig, y_pred):.4f}")

# Save model and feature columns
with open('model/lgb_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('model/feature_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

# ======================
# 9. Create Income Mapping from Training Data
# ======================
# Reload original data to get postcode (Locality) and municipality_income
df_full = pd.read_csv('data/sale_with_income.csv')
# Group by Locality and take mean income (should be constant per locality)
income_map = df_full.groupby('Locality')['municipality_income'].mean().to_dict()
with open('model/income_map.pkl', 'wb') as f:
    pickle.dump(income_map, f)

# Also save median income as fallback
median_income = df_full['municipality_income'].median()
with open('model/median_income.pkl', 'wb') as f:
    pickle.dump(median_income, f)

print("Income mapping and median income saved.")