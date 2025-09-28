import json
from model_utils import load_config, save_config  # Adjust import path if needed

def test_load_config_defaults(tmp_path):
    print("Running test_load_config_defaults")
    config_path = tmp_path / "config_test.json"
    config = load_config(str(config_path))
    assert config["model"] in ["arima", "lstm"]
    print("Finished test_load_config_defaults")

def test_save_config_creates_file(tmp_path):
    print("Running test_save_config_creates_file")
    config = {
        "model": "arima",
        "arima": {"p": 2, "d": 1, "q": 2, "use": True},
        "lstm": {"epochs": 10, "batch_size": 32, "sequence_length": 10, "use": False}
    }
    output_path = tmp_path / "test_output.json"
    save_config(config, output_path, username="tester")
    
    with open(output_path, "r") as f:
        saved = json.load(f)
    assert "meta" in saved and saved["meta"]["saved_by"] == "tester"
    print("Finished test_save_config_creates_file")
