import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import os
from model_utils import load_config

# Set page configuration
st.set_page_config(
    page_title="Hurricane Intensity Predictor",
    page_icon="🌪️",
    layout="wide"
)

# Use load_config for robustness (handles missing config2.json)
config = load_config()
model_type = config["model"]

st.title("🌪️ Hurricane Intensity Predictor")
st.markdown("> *\"Prediction is not just about numbers, it's about preparation.\"*")

# Create model directory if it doesn't exist
os.makedirs("models", exist_ok=True)

uploaded_file = st.file_uploader("📄 Upload your CSV file", type=["csv"])
forecast_steps = st.slider("⏱️ Forecast steps", 1, 30, 7)

if uploaded_file:
    try:
        # Treat 'NA' as missing values
        df = pd.read_csv(uploaded_file, na_values=["NA", "N/A", ""])
        
        # Display raw data
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        # Flexible wind speed column detection
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
            st.error("Missing wind speed column! Please upload a CSV with a column containing 'wind' in its name.")
            st.stop()
            
        # Clean wind_speed column: convert to numeric, fill missing
        df = df.rename(columns={wind_col: 'wind_speed'})
        df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')
        
        # Show data issues if any exist
        missing_count = df['wind_speed'].isna().sum()
        if missing_count > 0:
            st.warning(f"Found {missing_count} missing values in wind speed column. These will be filled using interpolation.")
            
        df['wind_speed'] = df['wind_speed'].fillna(method="ffill").fillna(method="bfill")
        if df['wind_speed'].isnull().any():
            st.error("Wind speed column still has missing values after cleaning! Please check your data.")
            st.stop()

        # Date handling
        if 'date' not in df.columns:
            date_col = None
            if all(c in df.columns for c in ['year', 'month', 'day']):
                st.info("Creating date column from year, month, and day columns")
                df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
                date_col = 'date'
            else:
                # Try to find any date-like column
                for col in df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            df['date'] = pd.to_datetime(df[col])
                            date_col = 'date'
                            st.info(f"Using {col} as date column")
                            break
                        except:
                            pass
                            
            if date_col is None:
                st.error("Missing date column! Please upload a CSV with a 'date' column or 'year', 'month', 'day' columns.")
                st.stop()
        
        # Ensure date is properly formatted
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if df['date'].isna().any():
                st.error("Some date values could not be converted to datetime format. Please check your data.")
                st.stop()
                
        # Sort by date and set as index
        df = df.sort_values('date').reset_index(drop=True)
        df_indexed = df.set_index('date')
        
        # Show cleaned data
        st.subheader("Cleaned Data")
        st.dataframe(df.head())
        
        # Plot historical data
        st.subheader("Historical Wind Speed")
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        ax_hist.plot(df['date'], df['wind_speed'])
        ax_hist.set_xlabel("Date")
        ax_hist.set_ylabel("Wind Speed")
        ax_hist.set_title("Historical Wind Speed")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_hist)

        # Forecasting section
        st.subheader("Forecast Results")
        
        try:
            if model_type == "arima":
                # Load ARIMA model
                try:
                    with open("models/arima_model.pkl", "rb") as f:
                        model = pickle.load(f)
                    forecast = model.forecast(steps=forecast_steps)
                    dates = pd.date_range(df['date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_steps)
                    results = pd.DataFrame({'date': dates, 'Forecast': forecast})
                except FileNotFoundError:
                    st.error("ARIMA model not found. Please train the model using the developer interface first.")
                    st.stop()
                    
            elif model_type == "lstm":
                # Load LSTM model
                try:
                    model = load_model("models/lstm_model.h5")
                    with open("models/lstm_scaler.pkl", "rb") as f:
                        scaler = pickle.load(f)
                    
                    seq_len = config['lstm']['sequence_length']
                    
                    if len(df) < seq_len:
                        st.error(f"Not enough data for LSTM prediction. Need at least {seq_len} data points.")
                        st.stop()
                        
                    scaled = scaler.transform(df['wind_speed'].values.reshape(-1, 1))
                    history = scaled[-seq_len:].flatten()
                    
                    predictions = []
                    for _ in range(forecast_steps):
                        x = np.array(history[-seq_len:]).reshape(1, seq_len, 1)
                        pred = model.predict(x, verbose=0)
                        predictions.append(pred[0][0])
                        history = np.append(history, pred[0][0])
                    
                    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                    dates = pd.date_range(df['date'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_steps)
                    results = pd.DataFrame({'date': dates, 'Forecast': forecast})
                except FileNotFoundError:
                    st.error("LSTM model not found. Please train the model using the developer interface first.")
                    st.stop()
            else:
                st.error(f"Unknown model type: {model_type}")
                st.stop()
                
            # Display results
            st.dataframe(results)
            
            # Plot forecast with historical data
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['date'], df['wind_speed'], label="Historical Wind Speed")
            ax.plot(results['date'], results['Forecast'], label="Forecast", marker="o", color='red')
            ax.set_xlabel("Date")
            ax.set_ylabel("Wind Speed")
            ax.set_title("Wind Speed Forecast")
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Download buttons
            col1, col2 = st.columns(2)
            
            # Download forecast as CSV
            csv = results.to_csv(index=False).encode('utf-8')
            col1.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name='forecast_results.csv',
                mime='text/csv'
            )

            # Download plot as PNG
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300)
            buf.seek(0)
            col2.download_button(
                label="Download Forecast Plot",
                data=buf.getvalue(),
                file_name="forecast_plot.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            st.exception(e)
            
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        st.exception(e)
else:
    st.info("Please upload a CSV file with wind speed data to generate a forecast.")
    
# Show information about expected data format
with st.expander("ℹ️ Input Data Format"):
    st.markdown("""
    The CSV file should contain:
    
    1. A column containing wind speed data (with 'wind' in the column name)
    2. Date information in one of these formats:
       - A 'date' column 
       - Separate 'year', 'month', and 'day' columns
    
    Example data format:
    | date       | wind_speed |
    |------------|------------|
    | 2023-01-01 | 45         |
    | 2023-01-02 | 52         |
    
    or:
    
    | year | month | day | wind |
    |------|-------|-----|------|
    | 2023 | 1     | 1   | 45   |
    | 2023 | 1     | 2   | 52   |
    """)