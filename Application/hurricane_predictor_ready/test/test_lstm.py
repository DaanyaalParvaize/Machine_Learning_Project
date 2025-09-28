import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from model_utils import save_lstm_model, load_lstm_model  # adjust if needed

def test_lstm_model_save_and_load(tmp_path):
    print("Running test_lstm_model_save_and_load")
    model = Sequential([
        LSTM(10, input_shape=(10, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    scaler = MinMaxScaler()
    dummy_data = np.random.rand(100, 1)
    scaler.fit(dummy_data)
    
    model_path = tmp_path / "test_lstm_model.h5"
    scaler_path = tmp_path / "test_scaler.pkl"
    save_lstm_model(model, scaler, str(model_path), str(scaler_path))
    
    loaded_model, loaded_scaler = load_lstm_model(str(model_path), str(scaler_path))
    assert loaded_model is not None
    assert loaded_scaler is not None
    print("Finished test_lstm_model_save_and_load")
