# 🌪️ Hurricane Intensity Prediction using ARIMA and LSTM
# 📂 hurricane-intensity-prediction/
│
├── app.py                   # Streamlit user interface
├── VAR.py                   # Vector autoregression (for comparison)
├── training.py               # Model training logic
├── plot.py                   # Visualization and graph functions
├── data/                     # Dataset folder (NOAA data)
├── images/                   # Output graphs and figures
├── reports/                  # LaTeX docs and presentation files
├── requirements.txt           # Required libraries
└── README.md                  # Project documentation
## 📘 Overview
This project aims to **predict hurricane wind speeds** using time series forecasting models — **ARIMA** (AutoRegressive Integrated Moving Average) and **LSTM** (Long Short-Term Memory Neural Network).  
It utilizes the **NOAA Atlantic Hurricane Database (1975–2021)**, available on Kaggle, to analyze and forecast hurricane intensity patterns.

The project supports:
- 🌍 **Disaster Preparedness** – Early warning systems for emergency planning  
- 🏗️ **Infrastructure Design** – Safer and more resilient construction planning  
- 💰 **Insurance & Risk Management** – Improved damage estimation and premium modeling  
- 🔬 **Scientific Research** – Deeper understanding of climatic patterns and storm behaviors  

---

## 🧠 Objectives
- Analyze historical hurricane data from NOAA  
- Build ARIMA and LSTM models for forecasting hurricane intensity  
- Compare statistical vs. deep learning approaches  
- Deploy an interactive forecasting interface using **Streamlit**

---

## 🧰 Technologies Used
- **Programming Language:** Python 3.9+  
- **Framework:** Streamlit  
- **Libraries:**
  - Pandas, NumPy – Data handling and preprocessing  
  - Matplotlib, Seaborn – Visualization  
  - Statsmodels – ARIMA modeling  
  - TensorFlow / Keras – LSTM modeling  
  - Scikit-learn – Data scaling and evaluation  

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/hurricane-intensity-prediction.git
   cd hurricane-intensity-prediction

## Optional 
python -m venv env
source env/bin/activate     # On Mac/Linux
env\Scripts\activate        # On Windows

# Install Dependencies 
pip install -r requirements.txt

# Launch the Streamlit application

streamlit run app.py
Upload your dataset (CSV format containing date and wind speed columns).

Choose the model — ARIMA or LSTM.

Set the date range for training and testing.

The app will:

Train the model

Predict future wind speeds

Display visual comparisons (Forecast vs. Actuals)





