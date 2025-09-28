import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt

##
# @brief Generates a synthetic dataset of dates and wind speeds.
# @return pandas.DataFrame with columns 'date' and 'wind_speed'.
##
dates = pd.date_range(start='2023-01-01', periods=20)
wind_speeds = [45, 48, 52, 50, 55, 60, 58, 62, 65, 70, 68, 66, 64, 60, 58, 55, 50, 48, 45, 42]
df = pd.DataFrame({'date': dates, 'wind_speed': wind_speeds})

##
# @brief Forecast future wind speeds using an ARIMA model.
# @param data pandas.DataFrame containing 'wind_speed' and 'date' columns.
# @param order tuple ARIMA order parameters (p, d, q), default is (2, 1, 2).
# @param steps int Number of future time steps to forecast, default is 5.
# @return pandas.DataFrame with future 'date' and ARIMA 'Forecast' values.
##
def forecast_arima(data, order=(2, 1, 2), steps=5):
    model = ARIMA(data['wind_speed'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    future_dates = pd.date_range(start=data['date'].iloc[-1] + pd.Timedelta(days=1), periods=steps)
    return pd.DataFrame({'date': future_dates, 'Forecast': forecast})

##
# @brief Forecast future wind speeds using a pre-trained LSTM model.
# @param data pandas.DataFrame containing 'wind_speed' and 'date' columns.
# @param model_path str Path to the saved LSTM Keras model (.h5 file).
# @param scaler_path str Path to the saved MinMaxScaler pickle file.
# @param steps int Number of future time steps to forecast, default is 5.
# @param seq_len int Length of input sequence for the LSTM model, default is 10.
# @return pandas.DataFrame with future 'date' and LSTM 'Forecast' values.
##
def forecast_lstm(data, model_path='models/lstm_model.h5', scaler_path='models/lstm_scaler.pkl', steps=5, seq_len=10):
    # Load the trained LSTM model
    model = load_model(model_path)
    # Load the scaler used during training
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    # Scale the wind speed data
    scaled = scaler.transform(data['wind_speed'].values.reshape(-1, 1))
    # Initialize the input sequence with the last seq_len scaled values
    history = scaled[-seq_len:].flatten()
    predictions = []
    for _ in range(steps):
        # Prepare input of shape (1, seq_len, 1)
        x = np.array(history[-seq_len:]).reshape(1, seq_len, 1)
        # Predict the next step
        pred = model.predict(x, verbose=0)
        predictions.append(pred[0][0])
        # Append prediction to history for next input
        history = np.append(history, pred[0][0])
    # Inverse scale the predicted values to original scale
    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    # Create future date range for the forecast
    future_dates = pd.date_range(start=data['date'].iloc[-1] + pd.Timedelta(days=1), periods=steps)
    return pd.DataFrame({'date': future_dates, 'Forecast': forecast})

# Execute ARIMA forecast
arima_results = forecast_arima(df)

# Execute LSTM forecast
lstm_results = forecast_lstm(df)

##
# @brief Visualizes historical wind speeds and forecasts from ARIMA and LSTM.
##
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['wind_speed'], label='Historical Wind Speed')
plt.plot(arima_results['date'], arima_results['Forecast'], label='ARIMA Forecast', marker='o')
plt.plot(lstm_results['date'], lstm_results['Forecast'], label='LSTM Forecast', marker='x')
plt.xlabel('Date')
plt.ylabel('Wind Speed (mph)')
plt.title('Hurricane Wind Speed Forecasting')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('forecast_comparison.png')
plt.close()
