import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats  # For outlier removal
from model_utils import load_config, save_config  # Import config functions

# --------- Utils ---------

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

    fixed_series = pd.Series(fixed_dates).fillna(pd.Timestamp.today())
    return fixed_series

def load_storm_data(file_input):
    # file_input can be filepath string or Streamlit UploadedFile
    if isinstance(file_input, str):
        if not os.path.exists(file_input):
            st.error(f"Data file not found at {file_input}")
            return None
        df = pd.read_csv(file_input, na_values=["NA", "N/A", ""])
    else:
        file_input.seek(0)
        df = pd.read_csv(file_input, na_values=["NA", "N/A", ""])

    # Flexible date handling
    if 'date' not in df.columns:
        if all(c in df.columns for c in ['year', 'month', 'day']):
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        else:
            found_date_col = False
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        df['date'] = fix_and_parse_dates(df[col])
                        found_date_col = True
                        break
                    except Exception:
                        continue
            if not found_date_col:
                st.error("CSV must contain 'date' or 'year','month','day' columns")
                return None
    else:
        df['date'] = fix_and_parse_dates(df['date'])

    # Wind speed column detection
    wind_col = None
    for c in df.columns:
        if 'wind' in c.lower() and 'speed' in c.lower():
            wind_col = c
            break
    if wind_col is None:
        for c in df.columns:
            if 'wind' in c.lower():
                wind_col = c
                break
    if wind_col is None:
        st.error("CSV must contain a wind speed column (name with 'wind')")
        return None
    df.rename(columns={wind_col: 'wind_speed'}, inplace=True)
    df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')
    df['wind_speed'].fillna(method='ffill', inplace=True)
    df['wind_speed'].fillna(method='bfill', inplace=True)

    # ======= Outlier removal =======
    z_scores = np.abs(stats.zscore(df['wind_speed']))
    outliers = z_scores > 3
    num_outliers = outliers.sum()
    if num_outliers > 0:
        st.warning(f"Removed {num_outliers} outlier wind speed values based on z-score > 3")
        df = df.loc[~outliers].reset_index(drop=True)
    # ======= End outlier removal =======

    df = df.sort_values('date').reset_index(drop=True)
    return df

def save_arima_model(model_fit, path='models/arima_model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model_fit, f)

def save_lstm_model(model, scaler, model_path='models/lstm_model.h5', scaler_path='models/lstm_scaler.pkl'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

# --------- Model Training Functions ---------

def train_arima_model(train_data, p, d, q):
    values = train_data['wind_speed'].values
    z_scores = np.abs(stats.zscore(values))
    filtered_values = values[z_scores <= 3]
    if len(filtered_values) < 10:
        raise ValueError("Not enough data after outlier removal for ARIMA training.")
    model = ARIMA(filtered_values, order=(p, d, q))
    model_fit = model.fit()
    return model_fit

def calculate_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return rmse, mae

def train_lstm_model(train_data, epochs, batch_size, sequence_length):
    values = train_data['wind_speed'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)

    if len(scaled_values) <= sequence_length:
        raise ValueError(f"Not enough data for LSTM with sequence length {sequence_length}")

    X, y = [], []
    for i in range(sequence_length, len(scaled_values)):
        X.append(scaled_values[i-sequence_length:i, 0])
        y.append(scaled_values[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(units=50),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    return model, scaler, X, y

def lstm_predict(model, scaler, train_data, sequence_length):
    scaled_values = scaler.transform(train_data['wind_speed'].values.reshape(-1, 1))
    history_seq = list(scaled_values[-sequence_length:].flatten())

    forecast_steps = min(10, len(scaled_values))
    predictions = []

    for _ in range(forecast_steps):
        X_input = np.array(history_seq[-sequence_length:]).reshape(1, sequence_length, 1)
        pred = model.predict(X_input, verbose=0)
        predictions.append(pred[0][0])
        history_seq.append(pred[0][0])

    predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predicted_values

# --------- Streamlit UI ---------

def main():
    st.title("🛠️ Developer Training Interface for Hurricane Intensity Predictor")
    st.write("Upload your storm data CSV, set model parameters, and train models here.")

    uploaded_file = st.file_uploader("Upload CSV file with storm data", type=["csv"])

    if uploaded_file is not None:
        df = load_storm_data(uploaded_file)
        if df is None:
            st.stop()

        st.subheader("Data preview")
        st.dataframe(df.head())

        model_choice = st.selectbox("Select model to train", options=["arima", "lstm"])

        if model_choice == "arima":
            st.subheader("ARIMA parameters")
            p = st.number_input("AR term (p)", min_value=0, max_value=5, value=2)
            d = st.number_input("Difference term (d)", min_value=0, max_value=2, value=1)
            q = st.number_input("MA term (q)", min_value=0, max_value=5, value=2)

            if st.button("Train ARIMA model"):
                with st.spinner("Training ARIMA model..."):
                    try:
                        model_fit = train_arima_model(df, p, d, q)
                        st.success("ARIMA model trained successfully!")

                        st.subheader("ARIMA model summary")
                        st.text(model_fit.summary())

                        fitted_values = model_fit.fittedvalues
                        actual_values = pd.Series(df['wind_speed'].values[:len(fitted_values)])

                        rmse, mae = calculate_metrics(actual_values, fitted_values)
                        st.write(f"RMSE on training data: {rmse:.3f}")
                        st.write(f"MAE on training data: {mae:.3f}")

                        save_arima_model(model_fit)

                        # Update config after training ARIMA
                        config = load_config()
                        config['model'] = 'arima'
                        config['arima']['p'] = p
                        config['arima']['d'] = d
                        config['arima']['q'] = q
                        config['arima']['use'] = True
                        config['lstm']['use'] = False
                        save_config(config)
                        st.success("Config updated and saved!")

                    except Exception as e:
                        st.error(f"Error training ARIMA: {str(e)}")

        elif model_choice == "lstm":
            st.subheader("LSTM parameters")
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10)
            batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=32)
            sequence_length = st.number_input("Sequence length", min_value=1, max_value=20, value=10)

            if st.button("Train LSTM model"):
                with st.spinner("Training LSTM model... This may take a while."):
                    try:
                        model, scaler, X, y = train_lstm_model(df, epochs, batch_size, sequence_length)
                        st.success("LSTM model trained successfully!")

                        st.subheader("LSTM model summary")
                        from io import StringIO
                        stream = StringIO()
                        model.summary(print_fn=lambda x: stream.write(x + '\n'))
                        st.text(stream.getvalue())

                        predictions_scaled = model.predict(X)
                        predictions = scaler.inverse_transform(predictions_scaled).flatten()
                        actual = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

                        rmse, mae = calculate_metrics(actual, predictions)
                        st.write(f"RMSE on training data: {rmse:.3f}")
                        st.write(f"MAE on training data: {mae:.3f}")

                        save_lstm_model(model, scaler)

                        # Update config after training LSTM
                        config = load_config()
                        config['model'] = 'lstm'
                        config['lstm']['epochs'] = epochs
                        config['lstm']['batch_size'] = batch_size
                        config['lstm']['sequence_length'] = sequence_length
                        config['lstm']['use'] = True
                        config['arima']['use'] = False
                        save_config(config)
                        st.success("Config updated and saved!")

                    except Exception as e:
                        st.error(f"Error training LSTM: {str(e)}")

if __name__ == "__main__":
    main()
