import json
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from model_utils import load_config, load_storm_data, save_arima_model, save_lstm_model

def main():
    """
    Main function to train and save models based on configuration settings.
    Handles both ARIMA and LSTM models.
    """
    print("Starting model training...")
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Load configuration
    try:
        with open("config2.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Config file not found. Creating default config.")
        config = {
            "model": "arima",
            "arima": {"p": 2, "d": 1, "q": 2, "use": True},
            "lstm": {"epochs": 10, "batch_size": 32, "sequence_length": 10, "use": False}
        }
        with open("config2.json", "w") as f:
            json.dump(config, f, indent=4)
    
    # Get model choice from config
    model_choice = config.get("model", "arima").lower()
    print(f"Selected model: {model_choice}")
    
    # Load training data
    try:
        print("Loading training data...")
        train_data = load_storm_data("data/storms.csv")
        print(f"Loaded {len(train_data)} records")
        
        # Check if data is valid
        if train_data is None or len(train_data) < 10:
            print("Warning: Not enough training data. Using default data.")
            # Generate default data
            dates = pd.date_range(start='2023-01-01', periods=60)
            np.random.seed(42)
            wind_speeds = 50 + 15 * np.sin(np.arange(60) * 0.1) + np.random.normal(0, 5, 60)
            train_data = pd.DataFrame({
                'date': dates,
                'wind_speed': wind_speeds
            })
            # Save default data
            train_data.to_csv("data/storms.csv", index=False)
            print("Created default training data")
    
        # Ensure data is sorted by date and reset index
        train_data = train_data.sort_values('date').reset_index(drop=True)
        
        # Fill any missing values in wind_speed
        train_data['wind_speed'] = train_data['wind_speed'].fillna(method='ffill').fillna(method='bfill')
        
        # Train appropriate model based on configuration
        results_csv = os.path.join("models", "forecast_results.csv")
        results_png = os.path.join("models", "forecast_plot.png")
        
        if model_choice == "arima":
            train_arima(train_data, config, results_csv, results_png)
        elif model_choice == "lstm":
            train_lstm(train_data, config, results_csv, results_png)
        else:
            print(f"Unknown model type: {model_choice}")
            return
        
        print("Model training completed successfully!")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

def train_arima(train_data, config, results_csv, results_png):
    """Train ARIMA model and save results"""
    print("Training ARIMA model...")
    
    # Get ARIMA parameters from config
    p = config['arima'].get('p', 2)
    d = config['arima'].get('d', 1)
    q = config['arima'].get('q', 2)
    
    # Validate parameters
    p = max(0, min(5, p))  # Limit p between 0 and 5
    d = max(0, min(2, d))  # Limit d between 0 and 2
    q = max(0, min(5, q))  # Limit q between 0 and 5
    
    print(f"ARIMA parameters: p={p}, d={d}, q={q}")
    
    # Get wind speed values
    values = train_data["wind_speed"].values
    
    # Check if we have enough data
    if len(values) < 5:
        raise ValueError("Not enough data for ARIMA model. Need at least 5 rows.")
    
    # Create and fit ARIMA model
    model = ARIMA(values, order=(p, d, q))
    model_fit = model.fit()
    print("ARIMA model fitted successfully")
    
    # Calculate training errors (RMSE, MAE)
    predicted = model_fit.predict(start=0, end=len(values)-1)
    rmse = np.sqrt(mean_squared_error(values, predicted))
    mae = mean_absolute_error(values, predicted)
    print(f"ARIMA Model Training Errors — RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Generate forecast
    forecast_steps = min(10, len(values))  # forecast up to 10 steps or less if data is small
    forecast = model_fit.forecast(steps=forecast_steps)
    print(f"Forecasted wind speed for next {forecast_steps} steps: {forecast}")
    
    # Save model
    save_arima_model(model_fit, os.path.join("models", "arima_model.pkl"))
    print("ARIMA model saved")
    
    # Prepare results DataFrame
    last_date = pd.to_datetime(train_data['date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)
    results_df = pd.DataFrame({"date": future_dates, "forecasted_wind_speed": forecast})
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")
    
    # Plot and save results
    plt.figure(figsize=(10, 5))
    plt.plot(train_data['date'], train_data["wind_speed"], label="Historical Wind Speed")
    plt.plot(results_df["date"], results_df["forecasted_wind_speed"], label="Forecast", marker="o")
    plt.xlabel("Date")
    plt.ylabel("Wind Speed")
    plt.title(f"Wind Speed Forecast - ARIMA({p},{d},{q})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_png)
    plt.close()
    print(f"Forecast plot saved to {results_png}")

def train_lstm(train_data, config, results_csv, results_png):
    """Train LSTM model and save results"""
    print("Training LSTM model...")
    
    # Get LSTM parameters from config
    epochs = config['lstm'].get('epochs', 10)
    batch_size = config['lstm'].get('batch_size', 32)
    sequence_length = config['lstm'].get('sequence_length', 10)
    
    # Validate parameters
    epochs = max(1, min(100, epochs))  # Limit epochs between 1 and 100
    batch_size = max(1, min(128, batch_size))  # Limit batch_size between 1 and 128
    sequence_length = max(1, min(20, sequence_length))  # Limit sequence_length between 1 and 20
    
    print(f"LSTM parameters: epochs={epochs}, batch_size={batch_size}, sequence_length={sequence_length}")
    
    # Prepare data for LSTM
    values = train_data["wind_speed"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
    
    # Check if we have enough data
    if len(scaled_values) <= sequence_length:
        raise ValueError(f"Not enough data for LSTM. Need more than sequence_length={sequence_length} rows.")
    
    # Create sequences for LSTM
    X, y = [], []
    for i in range(sequence_length, len(scaled_values)):
        X.append(scaled_values[i-sequence_length:i, 0])
        y.append(scaled_values[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Create and train LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(units=50),
        Dense(units=1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    print("LSTM model compiled, starting training...")
    
    # Train the model
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)
    print("LSTM model trained successfully")
    
    # Calculate training errors (RMSE, MAE)
    train_pred_scaled = model.predict(X, verbose=0)
    train_pred = scaler.inverse_transform(train_pred_scaled).flatten()
    actual = scaler.inverse_transform(y.reshape(-1,1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(actual, train_pred))
    mae = mean_absolute_error(actual, train_pred)
    print(f"LSTM Model Training Errors — RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Save model and scaler
    save_lstm_model(model, scaler, 
                    os.path.join("models", "lstm_model.h5"), 
                    os.path.join("models", "lstm_scaler.pkl"))
    print("LSTM model and scaler saved")
    
    # Generate forecasts
    history_sequence = list(scaled_values[-sequence_length:].flatten())
    predictions = []
    forecast_steps = min(10, len(scaled_values))
    
    for _ in range(forecast_steps):
        X_input = np.array(history_sequence[-sequence_length:]).reshape(1, sequence_length, 1)
        pred = model.predict(X_input, verbose=0)
        predictions.append(pred[0][0])
        history_sequence.append(pred[0][0])
    
    # Inverse transform predictions
    predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    
    # Prepare results DataFrame
    last_date = pd.to_datetime(train_data['date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)
    results_df = pd.DataFrame({"date": future_dates, "forecasted_wind_speed": predicted_values})
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")
    
    # Plot and save results
    plt.figure(figsize=(10, 5))
    plt.plot(train_data['date'], train_data["wind_speed"], label="Historical Wind Speed")
    plt.plot(results_df["date"], results_df["forecasted_wind_speed"], label="Forecast", marker="o")
    plt.xlabel("Date")
    plt.ylabel("Wind Speed")
    plt.title("Wind Speed Forecast - LSTM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_png)
    plt.close()
    print(f"Forecast plot saved to {results_png}")

if __name__ == "__main__":
    main()
