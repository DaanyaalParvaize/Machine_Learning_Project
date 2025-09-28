import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from model_utils import train_arima_model, save_arima_model, save_lstm_model

def test_train_arima_from_notebook():
    print("Starting ARIMA training test")
    wind_data = np.random.normal(50, 5, 100)
    model = train_arima_model(wind_data, order=(2, 1, 2))
    path = save_arima_model(model, "models/test_arima_model.pkl")
    assert "test_arima_model.pkl" in path
    print("Finished ARIMA training test")

def test_train_lstm_from_notebook(tmp_path):
    print("Starting LSTM training test")
    wind_data = np.random.normal(60, 10, 120).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(wind_data)

    seq_len = 10
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i])
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=2, batch_size=16, verbose=0)

    model_path, scaler_path = save_lstm_model(
        model, scaler,
        str(tmp_path / "test_lstm_model.h5"),
        str(tmp_path / "test_scaler.pkl")
    )
    assert "test_lstm_model.h5" in model_path
    print("Finished LSTM training test")
