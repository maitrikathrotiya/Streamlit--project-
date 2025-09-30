import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

# ARIMA & SARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# LSTM (TensorFlow 2.x built-in Keras)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ------------------- STREAMLIT APP -------------------

st.title("📈 Nifty 50 Time Series Forecasting (ARIMA | SARIMA | LSTM)")

# ---- Load dataset ----
df = pd.read_csv("Nifty50.csv")

# ---- Clean columns ----
df.columns = df.columns.str.strip().str.lower()
df.rename(columns={"date": "date", "close": "close"}, inplace=True)

# Convert date
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Convert Close
df["close"] = df["close"].astype(str).str.replace(",", "", regex=False)
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df = df.dropna(subset=["close"])

# Sort
df = df.sort_values("date")

# ---- Show cleaned sample ----
st.success(f"✅ Dataset cleaned successfully! Total rows: {len(df)}")
st.dataframe(df.head())

# ---- Time series ----
ts = df.set_index("date")["close"]

# ---- Plot ----
fig = px.line(ts, y=ts.values, title="Nifty 50 Closing Price Over Time", labels={"y": "Close"})
st.plotly_chart(fig)

# ================================================================
# 1. ARIMA Forecast
# ================================================================
st.subheader("🔮 ARIMA Forecast (Next 12 Days)")
try:
    model_arima = ARIMA(ts, order=(5, 1, 0))
    model_arima_fit = model_arima.fit()
    forecast_arima = model_arima_fit.forecast(steps=12)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts.index, ts, label="Historical")
    ax.plot(forecast_arima.index, forecast_arima, label="Forecast", color="red")
    ax.set_title("ARIMA Forecast")
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.error(f"⚠️ ARIMA error: {e}")

# ================================================================
# 2. SARIMA Forecast
# ================================================================
# ✅ Corrected SARIMA Forecast Plot


st.subheader("🔮 SARIMA Forecast (Next 12 Days)")
try:
    model_sarima = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
    model_sarima_fit = model_sarima.fit(disp=False)
    forecast_sarima = model_sarima_fit.forecast(steps=12)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts.index, ts, label="Historical")
    ax.plot(forecast_sarima.index, forecast_sarima, label="Forecast", color="green")
    ax.set_title("SARIMA Forecast")
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.error(f"⚠️ SARIMA error: {e}")

# ================================================================
# 3. LSTM Forecast
# ================================================================
st.subheader("🔮 LSTM Forecast (Next 12 Days)")
try:
    # Prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))

    window = 10
    X, y = [], []
    for i in range(len(ts_scaled) - window):
        X.append(ts_scaled[i:i + window])
        y.append(ts_scaled[i + window])
    X, y = np.array(X), np.array(y)

    # Split train/test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build model
    model = Sequential()
    model.add(LSTM(50, activation="tanh", input_shape=(window, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    # Train
    model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=0)

    # Predict future
       # Predict future
    last_window = ts_scaled[-window:].reshape(1, window, 1)
    preds = []
    for _ in range(12):
         pred = model.predict(last_window, verbose=0)
         preds.append(pred[0][0])
    # FIXED concatenation
         pred_reshaped = pred.reshape(1, 1, 1)
         last_window = np.concatenate([last_window[:, 1:, :], pred_reshaped], axis=1)

    

    forecast_lstm = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    # Plot
    future_dates = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=12, freq="D")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts.index, ts, label="Historical")
    ax.plot(future_dates, forecast_lstm, label="Forecast", color="orange")
    ax.set_title("LSTM Forecast")
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"⚠️ LSTM error: {e}")
