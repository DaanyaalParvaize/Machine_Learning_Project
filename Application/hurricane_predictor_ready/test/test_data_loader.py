import pandas as pd
from model_utils import load_storm_data  # Adjust import path if needed

def test_generate_demo_data(tmp_path):
    print("Running test_generate_demo_data")
    demo_file = tmp_path / "storms.csv"
    df = load_storm_data(str(demo_file))
    assert "wind_speed" in df.columns
    assert "date" in df.columns
    assert df["wind_speed"].notnull().all()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    print("Finished test_generate_demo_data")

def test_read_valid_csv(tmp_path):
    print("Running test_read_valid_csv")
    test_csv = tmp_path / "storms.csv"
    pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=3),
        "wind_speed": [50, 52, 48]
    }).to_csv(test_csv, index=False)

    df = load_storm_data(str(test_csv))
    assert len(df) == 3
    assert pd.api.types.is_integer_dtype(df["wind_speed"]) or pd.api.types.is_float_dtype(df["wind_speed"])
    print("Finished test_read_valid_csv")
