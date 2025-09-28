import os
import json
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA

# Ensure we have the script directory for config path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config2.json')

def load_storm_data(file_path='data/storms.csv'):
    """
    Loads storm data from a CSV file, automatically detecting the wind speed column.
    Returns a DataFrame with a standardized 'wind_speed' column.
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Check if file exists
        if not os.path.exists(file_path):
            # If demo file doesn't exist, create a simple demo dataset
            start_date = '2023-01-01'
            end_date = '2023-03-01'
            dates = pd.date_range(start=start_date, end=end_date)
            np.random.seed(42)  # For reproducibility
            wind_speeds = np.random.normal(50, 15, size=len(dates))
            wind_speeds = np.maximum(wind_speeds, 0)  # Ensure no negative wind speeds
            
            demo_data = pd.DataFrame({
                'date': dates,
                'wind_speed': wind_speeds
            })
            
            # Save demo data
            demo_data.to_csv(file_path, index=False)
            print(f"Created demo dataset at {file_path}")
            return demo_data
            
        storm_data = pd.read_csv(file_path, na_values=["NA", "N/A", ""])

        # Auto-create date if missing
        if 'date' not in storm_data.columns:
            if all(col in storm_data.columns for col in ['year', 'month', 'day']):
                storm_data['date'] = pd.to_datetime(storm_data[['year', 'month', 'day']])
            else:
                # Try to find any date-like column
                date_col = None
                for col in storm_data.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            storm_data['date'] = pd.to_datetime(storm_data[col])
                            date_col = col
                            break
                        except:
                            pass
                
                if date_col is None:
                    raise ValueError("CSV must contain 'date' or 'year'/'month'/'day' columns")

        # Flexible wind speed column detection
        wind_col = None
        col_candidates = list(storm_data.columns)
        
        # First, look for columns containing both 'wind' and 'speed'
        for c in col_candidates:
            if 'wind' in c.lower() and 'speed' in c.lower():
                wind_col = c
                break
                
        # If not found, look for any column with 'wind'
        if wind_col is None:
            for c in col_candidates:
                if 'wind' in c.lower():
                    wind_col = c
                    break
                    
        if wind_col is None:
            raise ValueError("CSV must contain a wind speed column (e.g., 'wind_speed', 'Wind Speed', 'wind', etc.)")

        # Standardize column name and clean data
        storm_data = storm_data.rename(columns={wind_col: 'wind_speed'})
        
        # Convert to numeric and handle missing values
        storm_data['wind_speed'] = pd.to_numeric(storm_data['wind_speed'], errors='coerce')
        storm_data['wind_speed'] = storm_data['wind_speed'].fillna(method="ffill").fillna(method="bfill")
        
        # Ensure date is datetime and sort
        storm_data['date'] = pd.to_datetime(storm_data['date'])
        storm_data = storm_data.sort_values('date').reset_index(drop=True)

        return storm_data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File at {file_path} not found.")
    except Exception as e:
        raise Exception(f"Error loading storm data: {str(e)}")

def save_arima_model(model, path='models/arima_model.pkl'):
    """Save ARIMA model to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return path

def save_lstm_model(model, scaler, model_path='models/lstm_model.h5', scaler_path='models/lstm_scaler.pkl'):
    """Save LSTM model and scaler to disk"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    return model_path, scaler_path

def load_lstm_model(model_path='models/lstm_model.h5', scaler_path='models/lstm_scaler.pkl'):
    """Load LSTM model and scaler from disk"""
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model files not found. Please train the model first.")
        
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def load_config(config_path=CONFIG_PATH):
    """Load configuration from JSON file or return default if not found"""
    if not os.path.exists(config_path):
        default_config = {
            "model": "arima",
            "arima": {"p": 2, "d": 1, "q": 2, "use": True},
            "lstm": {"epochs": 10, "batch_size": 32, "use": False, "sequence_length": 10}
        }
        # Save default config
        save_config(default_config, config_path)
        return default_config

    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config, config_path=CONFIG_PATH, username=None):
    """Save configuration to JSON file"""
    config['meta'] = {
        'last_saved': datetime.now().isoformat(),
        'saved_by': username or 'unknown'
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return config_path

def train_arima_model(data, order=(2, 1, 2)):
    """Train an ARIMA model with the given order on the provided data"""
    if len(data) < 10:
        raise ValueError(f"Not enough data points for ARIMA model training. Got {len(data)}, need at least 10.")
        
    try:
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        return fitted_model
    except Exception as e:
        raise Exception(f"Error training ARIMA model: {str(e)}")