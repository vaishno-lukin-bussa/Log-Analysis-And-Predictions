import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
import itertools
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras import Input
from sklearn.preprocessing import StandardScaler
from datetime import datetime

st.set_page_config(page_title="Log Forecast", layout="centered")
st.title("Log Analysis and Prediction")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "log_lstm_model_csv.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_csv.save")

log_file = st.file_uploader("Upload .csv, .log, or .txt log file", type=["csv", "log", "txt"])

if log_file:
    file_extension = log_file.name.split('.')[-1].lower()

    if file_extension == 'csv':
        df = pd.read_csv(log_file)
        required_cols = {"date", "level", "thread", "message"}
        if not required_cols.issubset(df.columns):
            st.error("CSV must contain columns: date, level, thread, message")
            st.stop()
        df["date"] = pd.to_datetime(df["date"])

    else:
        def parse_log_line(line):
            match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (\w+)\s*\[(.*?)\] - (.*)$", line)
            if match:
                dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
                level = match.group(2)
                thread = match.group(3)
                message = match.group(4)
                return dt, level, thread, message
            return None, None, None, None

        lines = log_file.read().decode("utf-8").splitlines()
        parsed = [parse_log_line(line.strip()) for line in lines]
        df = pd.DataFrame(parsed, columns=["date", "level", "thread", "message"]).dropna()

    df["Month"] = df["date"].dt.to_period("M")
    df["LogKey"] = df["level"] + " - " + df["message"]

    monthly_counts = df.groupby(["Month", "LogKey"]).size().unstack(fill_value=0).sort_index()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(monthly_counts)
    joblib.dump(scaler, SCALER_PATH)

    def create_sequences(data, window=3):
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i:i + window])
            y.append(data[i + window])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled, window=3)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    if not os.path.exists(MODEL_PATH):
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
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=100, batch_size=8, verbose=0)
        model.save(MODEL_PATH)
        st.success("Model trained and saved.")
    else:
        st.info("Using previously saved model.")

    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="mse")
    scaler = joblib.load(SCALER_PATH)

    def forecast_next(data, steps=7):
        output = []
        current = data[-3:].copy()
        for _ in range(steps):
            x = current.reshape((1, current.shape[0], current.shape[1]))
            y = model.predict(x, verbose=0)[0]
            noise = np.random.normal(0, 0.05, size=y.shape)
            y = y + noise
            output.append(y)
            current = np.vstack([current[1:], [y]])
        return np.array(output)

    forecast_scaled = forecast_next(scaled, steps=7)
    forecast = scaler.inverse_transform(forecast_scaled)

    max_vals = monthly_counts.max().values
    forecast = np.clip(forecast, 0, max_vals)

    last_month = monthly_counts.index[-1].to_timestamp()
    future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=7, freq='MS').to_period("M")
    forecast_df = pd.DataFrame(forecast, columns=monthly_counts.columns, index=future_months)

    for col in forecast_df.columns:
        if monthly_counts[col].sum() == 1:
            forecast_df[col] = 0

    error_cols = [col for col in monthly_counts.columns if col.startswith("ERROR")]
    if error_cols:
        st.subheader("Historical & Forecasted ERROR Logs")

        fig, ax = plt.subplots(figsize=(12, 6))
        color_cycle = itertools.cycle(mcolors.TABLEAU_COLORS.values())
        color_map = {}

        for col in error_cols:
            color = next(color_cycle)
            color_map[col] = color
            ax.plot(monthly_counts.index.astype(str), monthly_counts[col], label=f"Hist - {col}", color=color)
            ax.plot(forecast_df.index.astype(str), forecast_df[col], linestyle='dotted', label=f"Pred - {col}", color=color)

        ax.set_title("ERROR Log Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Count")
        ax.legend(fontsize="small", loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("No ERROR logs found to forecast.")

    st.subheader("Predict Trend for Log Pattern")
    search_term = st.text_input("Enter keyword to forecast")
    if st.button("Forecast This Pattern"):
        matched_cols = [col for col in monthly_counts.columns if search_term.lower() in col.lower()]
        if not matched_cols:
            st.warning("No match found.")
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            for col in matched_cols:
                color = color_map.get(col, None)
                ax.plot(monthly_counts.index.astype(str), monthly_counts[col], label=f"Hist - {col}", color=color)
                ax.plot(forecast_df.index.astype(str), forecast_df[col], linestyle='dotted', label=f"Pred - {col}", color=color)
            ax.set_title(f"Forecast for '{search_term}'")
            ax.set_xlabel("Month")
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.subheader("Historical + Forecasted Values Table")
            hist_part = monthly_counts[matched_cols].copy()
            forecast_part = forecast_df[matched_cols].copy()
            combined = pd.concat([hist_part, forecast_part])
            combined_display = combined.round(1).astype(int).reset_index().rename(columns={"index": "Month"})
            st.dataframe(combined_display)
