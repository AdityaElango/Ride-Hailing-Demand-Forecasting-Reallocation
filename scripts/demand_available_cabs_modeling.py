# %% [markdown]
# Demand & Available Cabs — Modeling Notebook
#
# This notebook trains multiple models (LSTM, RandomForest, XGBoost, LightGBM, CatBoost, Prophet)
# to forecast **demand** and **available_cabs** from your dataset, compares evaluation metrics,
# selects the best model by a simple accuracy proxy, and uses it to forecast future values
# until a user-provided date (input format: `dd mm yyyy`).
#
# Notes / assumptions:
# - Your dataset path is used as provided: "C:\\Users\\adity\\Downloads\\MMDS\\Project\\Final_Processed.csv".
#   If not found, the notebook will try `/mnt/data/Final_Processed.csv` (this is where the platform may have placed it).
# - The notebook frames the forecasting problem as a supervised learning problem using lag features
#   (so tree-based models and LSTM can be used). Prophet uses its own API.
# - "Accuracy" is not well-defined for regression; we compute standard regression metrics (MAE, RMSE, MAPE, R2)
#   and a convenience 'accuracy' defined as: `max(0, 1 - MAE / mean(y_train)) * 100` — a relative measure.
# - GPU: where possible we enable GPU training (TensorFlow for LSTM, XGBoost with `gpu_hist`, LightGBM with `device='gpu'`, CatBoost `task_type='GPU`).
#   Make sure you have the correct CUDA/cuDNN and package builds installed for your Nvidia 4060.
# - This notebook is structured for use inside VS Code (open the file, run cells). It also works in Jupyter.

# %% [markdown]
# 1) Setup: packages & environment checks
# Run this cell to install missing packages. For GPU-enabled installs you may need to install system CUDA first; these pip commands assume appropriate GPU-supporting wheels are available.

# %%
# Install packages if you need to (uncomment to run). In VS Code, run in an environment where CUDA is installed for GPU acceleration.
# Note: Installing tensorflow with GPU support requires matching CUDA/cuDNN installed on the machine. If you don't have it, install `tensorflow` (CPU) instead.

# !pip install -U pip
# !pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm catboost prophet tensorflow==2.13.0  # pick a TF version compatible with your CUDA
# If prophet isn't found as `prophet`, try: pip install prophet

# %%
# Basic imports and GPU checks
import os
import sys
import math
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Try GPU-friendly libraries (import errors handled below)
try:
    import xgboost as xgb
except Exception as e:
    print('xgboost import failed:', e)

try:
    import lightgbm as lgb
except Exception as e:
    print('lightgbm import failed:', e)

try:
    from catboost import CatBoostRegressor
except Exception as e:
    print('catboost import failed:', e)

try:
    import tensorflow as tf
    tf_version = tf.__version__
    print('TensorFlow version', tf_version)
    gpus = tf.config.list_physical_devices('GPU')
    print('GPUs found by TF:', gpus)
except Exception as e:
    print('TensorFlow import failed:', e)

try:
    from prophet import Prophet
except Exception as e:
    print('Prophet import failed:', e)

# %% [markdown]
# 2) Load data
# Update `csv_path` if your CSV is at a different location.

# %%
csv_path_user = r"C:\Users\adity\Downloads\MMDS\Project\Final_Processed.csv"  # your provided path
csv_path_alt = '/mnt/data/Final_Processed.csv'

if os.path.exists(csv_path_user):
    csv_path = csv_path_user
elif os.path.exists(csv_path_alt):
    csv_path = csv_path_alt
else:
    raise FileNotFoundError(f"Dataset not found at either: {csv_path_user} or {csv_path_alt}. Please update path.")

print('Loading', csv_path)
df = pd.read_csv(csv_path)
print('Loaded rows:', len(df))

# Quick look
print(df.columns.tolist())
print(df.head())

# %% [markdown]
# 3) Preprocessing
# - Parse datetime
# - Sort by datetime
# - We'll build models for `demand` and `available_cabs` separately

# %%
# Parse datetime: adapt to your column name; you have `datetime`, `date`, `hour`, `time_zone` etc.
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])
elif 'date' in df.columns and 'hour' in df.columns:
    # combine date + hour
    df['datetime'] = pd.to_datetime(df['date'].astype(str)) + pd.to_timedelta(df['hour'].astype(int), unit='h')
else:
    raise ValueError('No datetime-like columns found')

# Sort
df = df.sort_values('datetime').reset_index(drop=True)

# Ensure target columns exist
for col in ['demand', 'available_cabs']:
    if col not in df.columns:
        raise ValueError(f'Missing {col} column in dataset')

# Optional — aggregate to hourly or daily depending on granularity.
# We'll keep the original frequency (assume hourly) — but if your data is per-ride,
# you should aggregate to the required frequency (e.g., hourly counts per pickup_area).

# If ride-level and you want total per timestamp across all pickup_area, uncomment and adapt:
# df_agg = df.groupby('datetime').agg({'demand':'sum','available_cabs':'sum'}).reset_index()
# df = df_agg.copy()

# Add time features
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour

df = df.set_index('datetime')

print('Data prepared — index range:', df.index.min(), 'to', df.index.max())

# %% [markdown]
# 4) Supervised framing (lag features)
# We'll create lag features for both targets and use them to predict one-step-ahead values.
# You can expand the horizon by iterating predictions.

# %%
LAGS = 24  # use last 24 hours as features — adjust as needed

def create_lag_features(df, target_col, lags=LAGS):
    df_feat = pd.DataFrame(index=df.index)
    df_feat[target_col] = df[target_col]
    for lag in range(1, lags+1):
        df_feat[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    # add time features
    df_feat['hour'] = df.index.hour
    df_feat['dayofweek'] = df.index.dayofweek
    df_feat['month'] = df.index.month
    df_feat = df_feat.dropna()
    return df_feat

# Create datasets for each target
df_demand = create_lag_features(df, 'demand', LAGS)
print('demand features shape:', df_demand.shape)

df_available = create_lag_features(df, 'available_cabs', LAGS)
print('available_cabs features shape:', df_available.shape)

# %% [markdown]
# 5) Train/test split (time-based). We'll keep the last 20% for testing.

# %%
def time_train_test_split(df_feat, test_size=0.2):
    n = len(df_feat)
    split = int(n * (1 - test_size))
    train = df_feat.iloc[:split]
    test = df_feat.iloc[split:]
    return train, test

train_demand, test_demand = time_train_test_split(df_demand, test_size=0.2)
train_avail, test_avail = time_train_test_split(df_available, test_size=0.2)

print('Train/test shapes demand:', train_demand.shape, test_demand.shape)

# Separate X/y
def split_xy(df_feat, target_col):
    X = df_feat.drop(columns=[target_col])
    y = df_feat[target_col]
    return X, y

X_train_d, y_train_d = split_xy(train_demand, 'demand')
X_test_d, y_test_d = split_xy(test_demand, 'demand')

X_train_a, y_train_a = split_xy(train_avail, 'available_cabs')
X_test_a, y_test_a = split_xy(test_avail, 'available_cabs')

# Keep column order
FEATURE_COLS_D = X_train_d.columns.tolist()
FEATURE_COLS_A = X_train_a.columns.tolist()

# %% [markdown]
# 6) Evaluation helpers

# %%

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100


def evaluate_regression(y_true, y_pred, y_train_mean):
    results = {}
    results['MAE'] = mean_absolute_error(y_true, y_pred)
    results['RMSE'] = rmse(y_true, y_pred)
    results['MAPE'] = mape(y_true, y_pred)
    results['R2'] = r2_score(y_true, y_pred)
    # simple relative 'accuracy' proxy
    results['accuracy'] = max(0.0, 1.0 - results['MAE'] / (y_train_mean if y_train_mean!=0 else 1)) * 100
    return results

# %% [markdown]
# 7) Train models helper functions

# %%
# Random Forest
from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train, X_test, seed=42):
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds

# XGBoost
def train_xgboost(X_train, y_train, X_test, use_gpu=True, seed=42):
    params = {'n_estimators':300, 'learning_rate':0.05, 'random_state':seed}
    if use_gpu:
        params.update({'tree_method':'gpu_hist', 'predictor':'gpu_predictor', 'gpu_id':0})
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds

# LightGBM
def train_lightgbm(X_train, y_train, X_test, use_gpu=True, seed=42):
    params = {'n_estimators':300, 'learning_rate':0.05, 'random_state':seed}
    if use_gpu:
        model = lgb.LGBMRegressor(**params, device='gpu')
    else:
        model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds

# CatBoost
def train_catboost(X_train, y_train, X_test, use_gpu=True, seed=42):
    params = {'iterations':1000, 'learning_rate':0.03, 'random_seed':seed, 'verbose':0}
    if use_gpu:
        params['task_type'] = 'GPU'
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds

# LSTM (TensorFlow)
def build_lstm_model(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae'])
    return model

# Prophet (per-target framing)
# Prophet expects a DataFrame with columns ds, y.

# %% [markdown]
# 8) Train & evaluate for each target across all models

# %%
import time
results = {'demand': {}, 'available_cabs': {}}

# Helper to run all models for a given target

def run_all_models(X_train, y_train, X_test, y_test, target_name):
    res = {}
    y_train_mean = float(np.mean(y_train))

    # RandomForest
    t0 = time.time()
    rf_model, rf_preds = train_random_forest(X_train, y_train, X_test)
    res['RandomForest'] = evaluate_regression(y_test, rf_preds, y_train_mean)
    res['RandomForest']['time_s'] = time.time() - t0

    # XGBoost
    try:
        t0 = time.time()
        xgb_model, xgb_preds = train_xgboost(X_train, y_train, X_test, use_gpu=True)
        res['XGBoost'] = evaluate_regression(y_test, xgb_preds, y_train_mean)
        res['XGBoost']['time_s'] = time.time() - t0
    except Exception as e:
        print('XGBoost training failed:', e)

    # LightGBM
    try:
        t0 = time.time()
        lgb_model, lgb_preds = train_lightgbm(X_train, y_train, X_test, use_gpu=True)
        res['LightGBM'] = evaluate_regression(y_test, lgb_preds, y_train_mean)
        res['LightGBM']['time_s'] = time.time() - t0
    except Exception as e:
        print('LightGBM training failed:', e)

    # CatBoost
    try:
        t0 = time.time()
        cat_model, cat_preds = train_catboost(X_train, y_train, X_test, use_gpu=True)
        res['CatBoost'] = evaluate_regression(y_test, cat_preds, y_train_mean)
        res['CatBoost']['time_s'] = time.time() - t0
    except Exception as e:
        print('CatBoost training failed:', e)

    # LSTM — needs reshaping and scaling
    try:
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()

        # reshape to (samples, timesteps, features) — we'll treat each sample as timesteps=1 and features=all
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        t0 = time.time()
        lstm = build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
        # Early stopping
        from tensorflow.keras.callbacks import EarlyStopping
        es = EarlyStopping(patience=5, restore_best_weights=True)
        lstm.fit(X_train_lstm, y_train_scaled, epochs=50, batch_size=64, validation_split=0.1, callbacks=[es], verbose=0)
        lstm_preds_scaled = lstm.predict(X_test_lstm).ravel()
        lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled.reshape(-1,1)).ravel()
        res['LSTM'] = evaluate_regression(y_test, lstm_preds, y_train_mean)
        res['LSTM']['time_s'] = time.time() - t0
    except Exception as e:
        print('LSTM failed:', e)

    # Prophet — reframe and run (one-step forecasting by training on history and predicting test dates)
    try:
        t0 = time.time()
        df_prophet = pd.DataFrame({'ds': y_train.index, 'y': y_train.values})
        m = Prophet()
        m.fit(df_prophet)
        future = pd.DataFrame({'ds': y_test.index})
        forecast = m.predict(future)
        prophet_preds = forecast['yhat'].values
        res['Prophet'] = evaluate_regression(y_test, prophet_preds, y_train_mean)
        res['Prophet']['time_s'] = time.time() - t0
    except Exception as e:
        print('Prophet failed:', e)

    return res

# Run for demand
print('\nRunning models for demand...')
results['demand'] = run_all_models(X_train_d, y_train_d, X_test_d, y_test_d, 'demand')

# Run for available_cabs
print('\nRunning models for available_cabs...')
results['available_cabs'] = run_all_models(X_train_a, y_train_a, X_test_a, y_test_a, 'available_cabs')

# %% [markdown]
# 9) Build results table and pick best model

# %%
# Convert results dict to DataFrame for each target
summary_tables = {}
for target in results:
    rows = []
    for model_name, metrics in results[target].items():
        row = {'model': model_name}
        row.update(metrics)
        rows.append(row)
    df_res = pd.DataFrame(rows)
    df_res = df_res.sort_values(by='accuracy', ascending=False).reset_index(drop=True)
    summary_tables[target] = df_res

# Show
print('\nResults — Demand')
print(summary_tables['demand'])
print('\nResults — Available Cabs')
print(summary_tables['available_cabs'])

# Save to CSV
summary_tables['demand'].to_csv('results_demand_comparison.csv', index=False)
summary_tables['available_cabs'].to_csv('results_available_cabs_comparison.csv', index=False)
print('Saved results CSVs to current directory')

# Pick best models (by accuracy)
best_model_names = {
    'demand': summary_tables['demand'].iloc[0]['model'],
    'available_cabs': summary_tables['available_cabs'].iloc[0]['model']
}
print('Best models by accuracy proxy:', best_model_names)

# %% [markdown]
# 10) Refit the chosen best models on the entire dataset (train+test) and forecast into the future up to user date

# %%
# Helper to train model object given model name (for reuse) — uses full data

def train_model_by_name(name, X_full, y_full):
    name = name.lower()
    if name == 'randomforest':
        m = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)
        m.fit(X_full, y_full)
        return m
    if name == 'xgboost':
        return train_xgboost(X_full, y_full, X_full, use_gpu=True)[0]
    if name == 'lightgbm':
        return train_lightgbm(X_full, y_full, X_full, use_gpu=True)[0]
    if name == 'catboost':
        return train_catboost(X_full, y_full, X_full, use_gpu=True)[0]
    if name == 'lstm':
        # train LSTM on full
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        Xs = scaler_X.fit_transform(X_full)
        ys = scaler_y.fit_transform(y_full.values.reshape(-1,1)).ravel()
        Xl = Xs.reshape((Xs.shape[0], 1, Xs.shape[1]))
        model = build_lstm_model((Xl.shape[1], Xl.shape[2]))
        from tensorflow.keras.callbacks import EarlyStopping
        es = EarlyStopping(patience=5, restore_best_weights=True)
        model.fit(Xl, ys, epochs=50, batch_size=64, validation_split=0.1, callbacks=[es], verbose=0)
        # Return a tuple containing model and scalers so we can inverse-transform later
        return (model, scaler_X, scaler_y)
    if name == 'prophet':
        df_prop = pd.DataFrame({'ds': X_full.index, 'y': y_full.values})
        m = Prophet()
        m.fit(df_prop)
        return m
    raise ValueError('Unknown model name: '+name)

# Rebuild full X/y for demand & available
full_demand = df_demand.copy()  # created earlier; make sure it's present
full_avail = df_available.copy()

X_full_d, y_full_d = split_xy(full_demand, 'demand')
X_full_a, y_full_a = split_xy(full_avail, 'available_cabs')

# Train best models
best_d_name = best_model_names['demand']
best_a_name = best_model_names['available_cabs']

print('Training best model for demand:', best_d_name)
best_d_model = train_model_by_name(best_d_name, X_full_d, y_full_d)
print('Training best model for available_cabs:', best_a_name)
best_a_model = train_model_by_name(best_a_name, X_full_a, y_full_a)

# %% [markdown]
# Forecasting routine: iterative multi-step forecasting until a target datetime provided by user (format `dd mm yyyy`)
# - We'll generate hourly timestamps from the last index to the requested date (end of day included).
# - For tree-based models, we iteratively build features using predicted lags.
# - For Prophet, we use its `predict` directly.

# %%

def forecast_until(model_obj, model_name, full_df_feat, target_col, end_date_str):
    # Parse end date
    end_dt = datetime.strptime(end_date_str, '%d %m %Y')
    last_ts = full_df_feat.index[-1]
    # generate hourly timestamps from last_ts + 1 hour to end_dt at hourly frequency
    periods = int(((end_dt - last_ts).total_seconds()) // 3600)
    if periods <= 0:
        raise ValueError('End date must be after last data timestamp: '+str(last_ts))

    future_index = [last_ts + timedelta(hours=i) for i in range(1, periods+1)]

    # start with a copy of the last LAGS rows for creating lag features
    df_work = full_df_feat.copy()

    # if LSTM model returns tuple, handle scalers
    is_lstm = isinstance(model_obj, tuple) and hasattr(model_obj[0], 'predict')

    preds = []
    for ts in future_index:
        # create feature row based on last LAGS values
        last_row = df_work.iloc[-LAGS:][target_col]
        if len(last_row) < LAGS:
            raise ValueError('Not enough history to create lags')
        feat = {}
        for i, val in enumerate(reversed(last_row.values), start=1):
            feat[f'{target_col}_lag_{i}'] = val
        # time features
        feat['hour'] = ts.hour
        feat['dayofweek'] = ts.dayofweek
        feat['month'] = ts.month
        X_feat = pd.DataFrame([feat])
        X_feat = X_feat[full_df_feat.drop(columns=[target_col]).columns]  # ensure column order

        # predict
        name = model_name.lower()
        if name in ['randomforest','xgboost','lightgbm','catboost']:
            pred = model_obj.predict(X_feat)[0]
        elif name == 'lstm':
            m, scaler_X, scaler_y = model_obj
            Xs = scaler_X.transform(X_feat)
            Xl = Xs.reshape((1,1,Xs.shape[1]))
            pred_scaled = m.predict(Xl).ravel()
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).ravel()[0]
        elif name == 'prophet':
            # prophet expects ds
            future_df = pd.DataFrame({'ds':[ts]})
            forecast = model_obj.predict(future_df)
            pred = forecast['yhat'].values[0]
        else:
            raise ValueError('Unknown model for forecasting: '+model_name)

        # append and add to df_work so next iterations use predicted lag
        preds.append((ts, pred))
        new_row = {}
        new_row[target_col] = pred
        for i in range(1, LAGS+1):
            # these are placeholders; actual lag columns will be created by create_lag_features if needed
            pass
        # append to df_work as a new row
        row_series = pd.Series(new_row, name=ts)
        df_work = pd.concat([df_work, pd.DataFrame({target_col:[pred]}, index=[ts])])

    pred_df = pd.DataFrame(preds, columns=['datetime', f'pred_{target_col}']).set_index('datetime')
    return pred_df

# %% [markdown]
# Example usage:
# - Provide end date in `dd mm yyyy` format to predict until that date (hourly predictions).
#
# end_date_str = '31 12 2025'
# preds_demand = forecast_until(best_d_model, best_d_name, full_demand, 'demand', end_date_str)
# preds_avail = forecast_until(best_a_model, best_a_name, full_avail, 'available_cabs', end_date_str)
#
# shortage = preds_demand['pred_demand'] - preds_avail['pred_available_cabs']
#
# Save results as CSV
# preds_demand.to_csv('future_preds_demand.csv')
# preds_avail.to_csv('future_preds_available_cabs.csv')
# shortage.to_csv('future_shortage.csv')

# %% [markdown]
# 11) Wrap-up notes
# - The forecasting routine is iterative and depends on the lag features; you may need to adapt LAGS, frequency, and aggregation.
# - If your dataset is very large, consider increasing `LAGS` or using more advanced sequence models.
# - For production forecasting consider ensembling multiple models or constructing prediction intervals.

# End of notebook
