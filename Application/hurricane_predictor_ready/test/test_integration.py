import numpy as np
import pandas as pd
from model_utils import load_config, save_config, train_arima_model  # Adjust import path if needed

def test_end_to_end_arima_training(tmp_path):
    print("Starting end-to-end ARIMA training test")
    config_path = tmp_path / "config.json"
    config = load_config(config_path)
    config["arima"]["p"] = 1
    config["arima"]["d"] = 1
    config["arima"]["q"] = 1
    save_config(config, config_path, username="test_runner")
    
    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=50),
        "wind_speed": np.random.normal(60, 10, 50)
    })
    
    model = train_arima_model(df["wind_speed"], (1, 1, 1))
    assert model is not None
    print("Completed end-to-end ARIMA training test")
