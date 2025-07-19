from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import Input
from sklearn.preprocessing import MinMaxScaler
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
def create_sequences(data, seq_len=3):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

def train_lstm_forecast(series, months=7, seq_len=3):
    if len(series) < seq_len + 2:
        return [series.mean()] * months

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = create_sequences(scaled, seq_len)
    X = X.reshape(-1, seq_len, 1)

    model = Sequential([
        Input(shape=(seq_len, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=140, verbose=0)

    forecasts = []
    input_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
    for _ in range(months):
        pred = model.predict(input_seq, verbose=0)
        forecasts.append(pred[0][0])
        input_seq = np.concatenate((input_seq[:, 1:, :], pred.reshape(1, 1, 1)), axis=1)


    return scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()

# --- 5. Forecast each error ---
forecast_results = {}
for col in pivoted.columns:
    forecast_results[col] = train_lstm_forecast(pivoted[col])

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
        for error_type, preds in forecast_results.items():
            forecast_rows.append({
                "timestamp": month.strftime("%Y-%m-%dT%H:%M:%S"),
                "error_type": error_type,
                "count": int(preds[i]),
                "type": "forecast"
            })

    return actual_rows + forecast_rows

# --- Run with: uvicorn main:app --reload ---
