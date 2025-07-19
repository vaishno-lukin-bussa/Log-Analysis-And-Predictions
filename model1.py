import streamlit as st
import pandas as pd
import numpy as np
import re
import itertools
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import Input
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.set_page_config(page_title="Log Forecasting with LSTM", layout="centered")
st.title("🔧 Predictive Maintenance - ERROR Forecasting")

# --- Upload Log File ---
log_file = st.file_uploader("Upload your log file", type=["log", "txt"])
if log_file:

    # --- Log Parser ---
    def parse_log_line(line):
        match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - (\w+)  \[(.*?)\] - (.*)$", line)
        if match:
            dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
            level = match.group(2)
            thread = match.group(3)
            message = match.group(4)
            return dt, level, thread, message
        return None, None, None, None

    lines = log_file.read().decode("utf-8").splitlines()
    parsed = [parse_log_line(line.strip()) for line in lines]
    log_df = pd.DataFrame(parsed, columns=["date", "level", "thread", "message"]).dropna()
    log_df["month"] = log_df["date"].dt.to_period("M")

    monthly_index = pd.period_range("2025-01", "2025-05", freq="M")
    future_months = pd.period_range("2025-06", "2025-12", freq="M")
    full_index = monthly_index.append(future_months)

    def prepare_sequences(series, steps=3):
        X, y = [], []
        for i in range(len(series) - steps):
            X.append(series[i:i + steps])
            y.append(series[i + steps])
        return np.array(X), np.array(y)

    # === COMBINED ERROR FORECAST CHART ===
    st.subheader("📉 Combined Forecast: All Distinct ERROR Messages")

    error_logs = log_df[log_df["level"] == "ERROR"]
    unique_error_messages = error_logs["message"].unique()

    color_cycle = itertools.cycle(mcolors.TABLEAU_COLORS.values())
    error_color_map = {}

    fig_all, ax_all = plt.subplots(figsize=(9, 5))

    for msg in unique_error_messages:
        monthly_counts = error_logs[error_logs["message"] == msg].groupby("month").size().reindex(monthly_index, fill_value=0)
        if monthly_counts.sum() < 2:
            continue  # Skip if only one message total (LSTM can't train)

        color = next(color_cycle)
        error_color_map[msg] = color

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(monthly_counts.values.reshape(-1, 1))
        X, y = prepare_sequences(scaled, steps=3)
        if X.shape[0] == 0:
            continue

        model = Sequential([
            Input(shape=(3, 1)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X.reshape(X.shape[0], X.shape[1], 1), y, epochs=100, verbose=0)

        input_seq = scaled[-3:].flatten()
        preds = []
        for _ in range(7):
            pred = model.predict(input_seq.reshape(1, 3, 1), verbose=0)
            preds.append(pred[0][0])
            input_seq = np.append(input_seq[1:], pred)

        forecast_vals = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten().round().astype(int)
        full_series = pd.concat([monthly_counts, pd.Series(forecast_vals, index=future_months)])

        label = msg[:30] + ("..." if len(msg) > 30 else "")
        ax_all.plot(full_index.astype(str)[:5], full_series[:5], label=label, color=color)           # historical
        ax_all.plot(full_index.astype(str)[5:], full_series[5:], linestyle='dotted', color=color)    # predicted

    ax_all.set_title("Forecasted ERROR Messages")
    ax_all.set_xlabel("Month")
    ax_all.set_ylabel("Count")
    ax_all.grid(True, linestyle='--', alpha=0.5)
    ax_all.legend(fontsize='small', loc='upper left')
    plt.xticks(rotation=45)
    st.pyplot(fig_all)

    # === INPUT MESSAGE FORECAST ===
    input_msg = st.text_input("🔎 Enter a log message to forecast its frequency:")
    if input_msg:
        st.subheader("📘 Forecast for Input Message")

        input_matches = log_df[log_df["message"].str.strip().str.lower() == input_msg.strip().lower()]
        msg_monthly = input_matches.groupby("month").size().reindex(monthly_index, fill_value=0)

        if msg_monthly.sum() < 2:
            msg_forecast_series = pd.Series([0] * 7, index=future_months)
            msg_full = pd.concat([msg_monthly, msg_forecast_series])

            fig_msg, ax_msg = plt.subplots(figsize=(6, 4))
            ax_msg.plot(msg_full.index.astype(str)[:5], msg_full[:5], label='Actual', color='blue')
            ax_msg.plot(msg_full.index.astype(str)[5:], msg_full[5:], label='Predicted', linestyle='dotted', color='gray')
            ax_msg.set_title("Forecast for Input Message")
            ax_msg.set_xlabel("Month")
            ax_msg.set_ylabel("Count")
            ax_msg.grid(True, linestyle='--', alpha=0.6)
            ax_msg.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig_msg)

            st.subheader("📋 Forecast Table for Input Message")
            forecast_table = pd.DataFrame({
                "Month": msg_full.index.astype(str),
                "Message Count": msg_full.values
            })
            st.dataframe(forecast_table.reset_index(drop=True), use_container_width=True)

        else:
            scaler = MinMaxScaler()
            scaled_msg = scaler.fit_transform(msg_monthly.values.reshape(-1, 1))
            X_msg, y_msg = prepare_sequences(scaled_msg, steps=3)

            model = Sequential([
                Input(shape=(3, 1)),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X_msg.reshape(X_msg.shape[0], X_msg.shape[1], 1), y_msg, epochs=100, verbose=0)

            input_seq = scaled_msg[-3:].flatten()
            msg_preds = []
            for _ in range(7):
                pred = model.predict(input_seq.reshape(1, 3, 1), verbose=0)
                msg_preds.append(pred[0][0])
                input_seq = np.append(input_seq[1:], pred)

            forecast_vals_msg = scaler.inverse_transform(np.array(msg_preds).reshape(-1, 1)).flatten().round().astype(int)
            msg_forecast_series = pd.Series(forecast_vals_msg, index=future_months)
            msg_full = pd.concat([msg_monthly, msg_forecast_series])

            # Plot
            fig_msg, ax_msg = plt.subplots(figsize=(6, 4))
            ax_msg.plot(msg_full.index.astype(str)[:5], msg_full[:5], label='Actual', color='blue')
            ax_msg.plot(msg_full.index.astype(str)[5:], msg_full[5:], label='Predicted', linestyle='dotted', color='blue')
            ax_msg.set_title("Forecast for Input Message")
            ax_msg.set_xlabel("Month")
            ax_msg.set_ylabel("Count")
            ax_msg.grid(True, linestyle='--', alpha=0.6)
            ax_msg.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig_msg)

            # Table
            st.subheader("📋 Forecast Table for Input Message")
            forecast_table = pd.DataFrame({
                "Month": msg_full.index.astype(str),
                "Message Count": msg_full.values
            })
            st.dataframe(forecast_table.reset_index(drop=True), use_container_width=True)
