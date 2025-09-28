import os
import json
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats  # For outlier removal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config2.json')

def fix_and_parse_dates(date_series):
    current_year = datetime.now().year
    fixed_dates = []
    for val in date_series:
        if pd.isna(val):
            fixed_dates.append(pd.NaT)
            continue
        val_str = str(val).strip()
        if not any(char.isdigit() for char in val_str[-5:]):
            val_str = f"{val_str} {current_year}"
        try:
            dt = pd.to_datetime(val_str, errors='raise', infer_datetime_format=True)
        except Exception:
            dt = pd.Timestamp.today()
        fixed_dates.append(dt)
    fixed_series = pd.Series(fixed_dates)
    return fixed_series.fillna(pd.Timestamp.today())

def load_storm_data(file_path='data/storms.csv'):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        dates = pd.date_range(start='2023-01-01', end='2023-03-01')
        wind = np.random.normal(50, 15, len(dates))
        df = pd.DataFrame({'date': dates, 'wind_speed': np.maximum(wind, 0)})
        df.to_csv(file_path, index=False)
        return df
    df = pd.read_csv(file_path, na_values=["NA", "N/A", ""])
    if 'date' not in df.columns:
        if all(col in df.columns for col in ['year', 'month', 'day']):
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        else:
            date_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            if date_col:
                df['date'] = fix_and_parse_dates(df[date_col])
            else:
                raise ValueError("CSV must contain a 'date' or 'year/month/day' columns.")
    else:
        df['date'] = fix_and_parse_dates(df['date'])

    wind_col = next((c for c in df.columns if 'wind' in c.lower()), None)
    if not wind_col:
        raise ValueError("Missing wind speed column in data.")
    df.rename(columns={wind_col: 'wind_speed'}, inplace=True)
    df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce').ffill().bfill()

    # ======= Outlier removal =======
    z_scores = np.abs(stats.zscore(df['wind_speed']))
    outliers = z_scores > 3
    if outliers.sum() > 0:
        df = df.loc[~outliers].reset_index(drop=True)
    # ======= End outlier removal =======

    return df.sort_values('date').reset_index(drop=True)

def save_arima_model(model, path='models/arima_model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return path

def save_lstm_model(model, scaler, model_path='models/lstm_model.h5', scaler_path='models/lstm_scaler.pkl'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    return model_path, scaler_path

def load_lstm_model(model_path='models/lstm_model.h5', scaler_path='models/lstm_scaler.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model not trained yet.")
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def load_config(config_path=CONFIG_PATH):
    if not os.path.exists(config_path):
        config = {
            "model": "lstm",  # Default model set to LSTM
            "arima": {"p": 2, "d": 1, "q": 2, "use": False},
            "lstm": {"epochs": 10, "batch_size": 32, "sequence_length": 10, "use": True}
        }
        save_config(config, config_path)
        return config
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, config_path=CONFIG_PATH, username=None):
    config['meta'] = {
        'last_saved': datetime.now().isoformat(),
        'saved_by': username or 'developer'
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return config_path

def train_arima_model(data, order=(2, 1, 2)):
    if len(data) < 10:
        raise ValueError("ARIMA requires at least 10 records.")
    model = ARIMA(data, order=order)
    return model.fit()

def log_forecast_metrics(model_name, actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    metrics = {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 2)}
    print(f"[{model_name}] RMSE: {metrics['rmse']}, MAE: {metrics['mae']}, MAPE: {metrics['mape']}")
    return metrics
