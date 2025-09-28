import numpy as np
from model_utils import train_arima_model  # Adjust if your file/module is named differently

def test_arima_model_training_success():
    print("Running test_arima_model_training_success")
    data = np.random.normal(50, 5, 100)
    model = train_arima_model(data, (2, 1, 2))
    assert model is not None
    assert "ARIMA" in model.summary().as_text()
    print("Finished test_arima_model_training_success")

def test_arima_training_failure_on_short_data():
    print("Running test_arima_training_failure_on_short_data")
    short_data = np.array([50, 52, 53])
    try:
        train_arima_model(short_data, (2, 1, 2))
    except ValueError as e:
        assert "Not enough data points" in str(e)
    print("Finished test_arima_training_failure_on_short_data")

