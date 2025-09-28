import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from model_utils import (
    load_config, save_config, load_storm_data, 
    train_arima_model, save_arima_model, save_lstm_model
)

# Set page configuration
st.set_page_config(
    page_title="Hurricane Model Configuration",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Developer Model Configuration")

# Check if data directory exists
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)

# Load configuration
config = load_config()

# Display current config
st.write("### Current Configuration:")
st.json(config)

# Sidebar for data upload
st.sidebar.header("Data Management")

# Option to upload training data
uploaded_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type=["csv"])
if uploaded_file is not None:
    try:
        # Save the uploaded file
        data_path = os.path.join(data_dir, "storms.csv")
        with open(data_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Saved training data to {data_path}")
        
        # Try to load and display preview
        storm_data = load_storm_data(data_path)
        st.sidebar.write("Data Preview:")
        st.sidebar.dataframe(storm_data.head())
    except Exception as e:
        st.sidebar.error(f"Error processing uploaded file: {str(e)}")

# Main content area
tabs = st.tabs(["Data Preview", "Model Configuration", "Training Results"])

with tabs[0]:
    st.header("Training Data Preview")
    try:
        storm_data = load_storm_data('data/storms.csv')
        st.write(f"Loaded {len(storm_data)} records")
        st.dataframe(storm_data)
        
        # Plot the data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(storm_data['date'], storm_data['wind_speed'])
        ax.set_title("Wind Speed Data")
        ax.set_xlabel("Date")
        ax.set_ylabel("Wind Speed")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error loading training data: {str(e)}")
        st.info("Please upload training data using the sidebar.")

with tabs[1]:
    st.header("Model Configuration")
    
    model_choice = st.radio("Select the Model to Configure", ("ARIMA", "LSTM"))
    
    if model_choice == "ARIMA":
        st.subheader("ARIMA Parameters")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("p (AR)", value=config['arima'].get('p', 2), min_value=0, max_value=5)
        with col2:
            d = st.number_input("d (I)", value=config['arima'].get('d', 1), min_value=0, max_value=2)
        with col3:
            q = st.number_input("q (MA)", value=config['arima'].get('q', 2), min_value=0, max_value=5)
            
        use_arima = st.checkbox("Use ARIMA for prediction", value=config['arima'].get('use', True))
        
        st.info("""
        ARIMA Parameter Guidelines:
        - p: Number of lag observations (AR term)
        - d: Degree of differencing (I term)
        - q: Size of moving average window (MA term)
        """)

    elif model_choice == "LSTM":
        st.subheader("LSTM Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Epochs", value=config['lstm'].get('epochs', 10), min_value=1)
            batch_size = st.number_input("Batch Size", value=config['lstm'].get('batch_size', 32), min_value=1)
        with col2:
            sequence_length = st.number_input("Sequence Length", value=config['lstm'].get('sequence_length', 10), min_value=1)
            
        use_lstm = st.checkbox("Use LSTM for prediction", value=config['lstm'].get('use', False))
        
        st.info("""
        LSTM Parameter Guidelines:
        - Epochs: Number of complete passes through the training dataset
        - Batch Size: Number of samples processed before the model is updated
        - Sequence Length: Number of past time steps used to predict the next step
        """)

    train_button = st.button("Train and Save Model")

with tabs[2]:
    st.header("Training Results")
    
    # This content will be populated when the train button is clicked

# Training logic
if train_button:
    try:
        # Load the training data
        storm_data = load_storm_data('data/storms.csv')
        
        if len(storm_data) < 10:
            st.error("Not enough training data. Please upload a CSV with at least 10 records.")
        else:
            with st.spinner(f"Training {model_choice} model..."):
                if model_choice == "ARIMA":
                    # Update configuration
                    config["model"] = "arima"
                    config["arima"].update({
                        "p": p,
                        "d": d,
                        "q": q,
                        "use": use_arima
                    })
                    # Ensure LSTM is marked as not in use, but preserve other LSTM params
                    config["lstm"]["use"] = False
                    
                    # Train ARIMA model
                    arima_model = train_arima_model(storm_data['wind_speed'], (p, d, q))
                    model_path = save_arima_model(arima_model)
                    
                    # Display results in the training results tab
                    with tabs[2]:
                        st.success(f"ARIMA model trained and saved to {model_path}!")
                        
                        # Display model summary
                        st.subheader("Model Summary")
                        st.text(str(arima_model.summary()))
                        
                        # Plot forecast vs actual for training data
                        st.subheader("Training Forecast")
                        
                        # Get predictions for the training data
                        predictions = arima_model.predict(start=p, end=len(storm_data)-1)
                        
                        # Create a dataframe for plotting
                        results_df = pd.DataFrame({
                            'date': storm_data['date'][p:],
                            'actual': storm_data['wind_speed'][p:],
                            'predicted': predictions
                        })
                        
                        # Calculate RMSE
                        rmse = np.sqrt(np.mean((results_df['actual'] - results_df['predicted'])**2))
                        
                        # Plot the results
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(results_df['date'], results_df['actual'], label='Actual')
                        ax.plot(results_df['date'], results_df['predicted'], label='Predicted', linestyle='--')
                        ax.set_title(f"ARIMA({p},{d},{q}) Training Results - RMSE: {rmse:.2f}")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Wind Speed")
                        ax.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)

                elif model_choice == "LSTM":
                    # Update configuration
                    config["model"] = "lstm"
                    config["lstm"].update({
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "use": use_lstm
                    })
                    # Ensure ARIMA is marked as not in use, but preserve other ARIMA params
                    config["arima"]["use"] = False
                    
                    # Data preparation for LSTM
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled = scaler.fit_transform(storm_data['wind_speed'].values.reshape(-1, 1))
                    
                    # Create sequences
                    X, y = [], []
                    for i in range(sequence_length, len(scaled)):
                        X.append(scaled[i-sequence_length:i, 0])
                        y.append(scaled[i, 0])
                    
                    X, y = np.array(X), np.array(y)
                    X = X.reshape((X.shape[0], X.shape[1], 1))
                    
                    # Create LSTM model
                    model = Sequential([
                        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                        LSTM(50),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    
                    # Train model
                    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
                    
                    # Save model
                    model_path, scaler_path = save_lstm_model(model, scaler)
                    
                    # Display results in the training results tab
                    with tabs[2]:
                        st.success(f"LSTM model trained and saved to {model_path}!")
                        
                        # Plot training history
                        st.subheader("Training History")
                        
                        # Plot loss
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(history.history['loss'], label='Training Loss')
                        ax.plot(history.history['val_loss'], label='Validation Loss')
                        ax.set_title("LSTM Training and Validation Loss")
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel("Loss")
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Generate predictions for the training data
                        train_predictions = []
                        
                        # Use the last sequence_length points of the original data
                        last_sequence = scaled[-sequence_length:].reshape(1, sequence_length, 1)
                        current_prediction = model.predict(last_sequence, verbose=0)[0][0]
                        train_predictions.append(current_prediction)
                        
                        # FIXED: Use a fixed number of predictions to avoid infinite loop
                        # Also adding print statements for debugging
                        num_predictions = min(50, len(scaled) - sequence_length)
                        print(f"Making {num_predictions} predictions")
                        
                        # Recursive prediction for a fixed number of points
                        for i in range(num_predictions - 1):  # -1 because we already have the first prediction
                            print(f"Prediction {i+1}/{num_predictions-1}")
                            
                            # Update the sequence with the new prediction - FIXED VERSION
                            prediction_3d = np.array([[[current_prediction]]])
                            
                            # Append the prediction to the sequence
                            last_sequence = np.append(last_sequence[:, 1:, :], prediction_3d, axis=1)
                            
                            # Predict the next point
                            current_prediction = model.predict(last_sequence, verbose=0)[0][0]
                            train_predictions.append(current_prediction)
                        
                        print(f"Total predictions made: {len(train_predictions)}")
                        
                        # Inverse transform to get original scale
                        train_predictions = scaler.inverse_transform(
                            np.array(train_predictions).reshape(-1, 1)
                        ).flatten()
                        
                        # Create dates for prediction period (limited to available dates)
                        available_dates = len(storm_data) - sequence_length
                        num_pred_to_use = min(len(train_predictions), available_dates)
                        pred_dates = storm_data['date'][sequence_length:sequence_length+num_pred_to_use]
                        
                        # Calculate RMSE for overlapping period
                        actual_values = storm_data['wind_speed'][sequence_length:sequence_length+num_pred_to_use]
                        predictions_to_compare = train_predictions[:num_pred_to_use]
                        rmse = np.sqrt(np.mean((actual_values - predictions_to_compare)**2))
                        
                        # Plot the results
                        st.subheader("Training Forecast")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(storm_data['date'], storm_data['wind_speed'], label='Historical Data')
                        ax.plot(pred_dates, predictions_to_compare, label='LSTM Predictions', linestyle='--')
                        ax.set_title(f"LSTM Training Results - RMSE: {rmse:.2f}")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Wind Speed")
                        ax.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # Save the updated configuration
                save_config(config)
                st.success("Configuration saved successfully!")
                
                # Update the configuration display
                st.write("### Updated Configuration:")
                st.json(config)
                
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        st.exception(e)