from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import Input
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import re
from datetime import datetime

# --- 1. Parse log lines ---
def parse_log_line(line):
    match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - (\w+)\s+\[.*?\] - (.*)', line)
    if match:
        timestamp_str, level, message = match.groups()
        try:
            dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            return dt, level.upper(), message.strip()
        except ValueError:
            return None, None, None
    return None, None, None

# --- 2. Read logs ---
data = []
with open("x.log", "r") as file:
    for line in file:
        ts, level, msg = parse_log_line(line)
        if ts and level == "ERROR":
            data.append([ts, msg])

df = pd.DataFrame(data, columns=["timestamp", "message"])
df = df[df["timestamp"].notnull()]
df["month"] = df["timestamp"].dt.to_period("M")

# --- 3. Monthly aggregation ---
monthly_errors = df.groupby(["month", "message"]).size().reset_index(name="count")
pivoted = monthly_errors.pivot(index="month", columns="message", values="count").fillna(0)

# --- 4. LSTM training ---
def train_lstm_forecast_multivariate(pivoted_df, months=7, window=3):
    if pivoted_df.shape[0] <= window:
        return pd.DataFrame(np.zeros((months, pivoted_df.shape[1])), columns=pivoted_df.columns)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivoted_df.values)

    def create_sequences(data, window):
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i:i+window])
            y.append(data[i+window])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, window)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.4),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(y.shape[1])
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, batch_size=8, verbose=0)

    def forecast_next(data, steps=months):
        output = []
        current = data[-window:].copy()
        for _ in range(steps):
            x = current.reshape((1, current.shape[0], current.shape[1]))
            y_pred = model.predict(x, verbose=0)[0]
            noise = np.random.normal(0, 0.05, size=y_pred.shape)
            y_pred = y_pred + noise
            output.append(y_pred)
            current = np.vstack([current[1:], [y_pred]])
        return np.array(output)

    forecast_scaled = forecast_next(scaled_data, months)
    forecast = scaler.inverse_transform(forecast_scaled)
    forecast_df = pd.DataFrame(forecast, columns=pivoted_df.columns)

    max_vals = pivoted_df.max().values
    forecast_df = forecast_df.clip(lower=0, upper=max_vals)

    for col in forecast_df.columns:
        if pivoted_df[col].sum() == 1:
            forecast_df[col] = 0

    return forecast_df

# --- 5. Forecast using new model ---
forecast_df = train_lstm_forecast_multivariate(pivoted)

# --- 6. FastAPI setup ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 7. Serve combined actual + forecast data ---
@app.get("/forecast/grafana")
def serve_combined_forecast():
    actual_rows = []
    for month, row in pivoted.iterrows():
        for error_type, count in row.items():
            actual_rows.append({
                "timestamp": month.to_timestamp().strftime("%Y-%m-%dT%H:%M:%S"),
                "error_type": error_type,
                "count": int(count),
                "type": "actual"
            })

    future_months = pd.date_range("2025-06-01", periods=7, freq="MS")
    forecast_rows = []
    for i, month in enumerate(future_months):
        for error_type in forecast_df.columns:
            forecast_rows.append({
                "timestamp": month.strftime("%Y-%m-%dT%H:%M:%S"),
                "error_type": error_type,
                "count": int(forecast_df.iloc[i][error_type]),
                "type": "forecast"
            })

    return actual_rows + forecast_rows

# --- Run with: uvicorn main:app --reload ---
