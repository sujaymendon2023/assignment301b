"""Streamlit dashboard that lets you explore prediction ranges, actuals, and exit strategies."""

import os
import sys
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data import MarketData  # noqa: E402

st.set_page_config(page_title="Prediction SLA Explorer", layout="wide")


def _prediction_window_slider(predictions: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    min_ts = predictions["timestamp"].min()
    max_ts = predictions["timestamp"].max()
    if max_ts - min_ts < timedelta(minutes=15):
        return min_ts, max_ts

    slider_end = max_ts - timedelta(minutes=15)
    window_start = st.slider(
        "Window start (15-minute range)",
        min_value=min_ts,
        max_value=slider_end,
        value=min_ts,
        step=timedelta(minutes=1),
        format="YYYY-MM-DD HH:mm",
    )
    return window_start, window_start + timedelta(minutes=15)


def _prediction_markers(window_df: pd.DataFrame) -> pd.DataFrame:
    min_row = window_df.nsmallest(1, "predictions").iloc[0]
    max_row = window_df.nlargest(1, "predictions").iloc[0]
    return pd.DataFrame(
        [
            {"timestamp": min_row["timestamp"], "predictions": min_row["predictions"], "label": "Min"},
            {"timestamp": max_row["timestamp"], "predictions": max_row["predictions"], "label": "Max"},
        ]
    )


def _find_prediction_after(predictions: pd.DataFrame, target: pd.Timestamp) -> Optional[pd.Series]:
    idx = predictions["timestamp"].searchsorted(target)
    if idx >= len(predictions):
        return None
    return predictions.iloc[idx]


def _build_strategy_row(
    label: str,
    signal: str,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_pred: float,
    exit_pred: float,
    entry_actual: float,
    exit_actual: float,
) -> dict:
    prediction_diff = exit_pred - entry_pred
    actual_diff = exit_actual - entry_actual
    profit = actual_diff if signal == "BUY" else -actual_diff
    return {
        "Strategy": label,
        "Signal": signal,
        "Entry Time": entry_time,
        "Exit Time": exit_time,
        "Entry Prediction": entry_pred,
        "Exit Prediction": exit_pred,
        "Entry Actual": entry_actual,
        "Exit Actual": exit_actual,
        "Pred. Diff": prediction_diff,
        "Actual Diff": actual_diff,
        "Profit/Loss": profit,
    }


data = MarketData()
window_start, window_end = _prediction_window_slider(data.predictions)
window_df = data.window(window_start, window_end)

if window_df.empty:
    st.warning("No predictions available inside the selected 15-minute window.")
    st.stop()

markers_df = _prediction_markers(window_df)
line = (
    alt.Chart(window_df)
    .mark_line()
    .encode(x="timestamp:T", y="predictions:Q")
    .properties(title="Predictions for the selected window")
)
point = (
    alt.Chart(markers_df)
    .mark_point(size=100)
    .encode(x="timestamp:T", y="predictions:Q", color="label:N", tooltip=["label", "predictions"])
)

st.altair_chart(line + point, use_container_width=True)

min_row = window_df.nsmallest(1, "predictions").iloc[0]
max_row = window_df.nlargest(1, "predictions").iloc[0]
ordered = sorted(
    [
        {"type": "min", "row": min_row},
        {"type": "max", "row": max_row},
    ],
    key=lambda item: item["row"]["timestamp"],
)
signal = "BUY" if ordered[0]["type"] == "min" else "SELL"

entry_row = ordered[0]["row"]
exit_row = ordered[1]["row"]
entry_actual_time, entry_actual_value = data.actual_at_or_after(entry_row["timestamp"])
exit_actual_time, exit_actual_value = data.actual_at_or_after(exit_row["timestamp"])

st.metric("Min prediction", f"{min_row['predictions']:.2f}", delta=f"Time {min_row['timestamp'].strftime('%H:%M')}")
st.metric("Max prediction", f"{max_row['predictions']:.2f}", delta=f"Time {max_row['timestamp'].strftime('%H:%M')}")

st.markdown(
    f"""
    **Prediction difference:** {abs(max_row['predictions'] - min_row['predictions']):.2f}  
    **Actual difference:** {exit_actual_value - entry_actual_value:.2f}  
    **Signal:** {signal} (based on {ordered[0]['type']} occurring first)
    """
)

st.markdown(
    f"""
    **Actual values at the prediction extremes:**  
    - {ordered[0]['type'].upper()} at {entry_actual_time.strftime('%Y-%m-%d %H:%M')} → {entry_actual_value:.2f}  
    - {ordered[1]['type'].upper()} at {exit_actual_time.strftime('%Y-%m-%d %H:%M')} → {exit_actual_value:.2f}
    """
)

strategies = []
strategies.append(
    _build_strategy_row(
        "Strategy 1 · Min/Max",
        signal,
        entry_row["timestamp"],
        exit_row["timestamp"],
        entry_row["predictions"],
        exit_row["predictions"],
        entry_actual_value,
        exit_actual_value,
    )
)

for offset in range(2, 6):
    target_time = entry_row["timestamp"] + timedelta(minutes=offset)
    future_pred = _find_prediction_after(data.predictions, target_time)
    if not future_pred:
        continue
    future_actual_time, future_actual_value = data.actual_at_or_after(future_pred["timestamp"])
    strategies.append(
        _build_strategy_row(
            f"Strategy 2 · Exit after {offset} min",
            signal,
            entry_row["timestamp"],
            future_pred["timestamp"],
            entry_row["predictions"],
            future_pred["predictions"],
            entry_actual_value,
            future_actual_value,
        )
    )

thresholds = [5, 7, 9]
future_preds = data.predictions[data.predictions["timestamp"] > entry_row["timestamp"]].copy()
future_preds["diff"] = (future_preds["predictions"] - entry_row["predictions"]).abs()
for threshold in thresholds:
    row = future_preds[future_preds["diff"] >= threshold].head(1)
    if row.empty:
        continue
    candidate = row.iloc[0]
    candidate_actual_time, candidate_actual_value = data.actual_at_or_after(candidate["timestamp"])
    strategies.append(
        _build_strategy_row(
            f"Strategy 3 · |Δ| ≥ {threshold}",
            signal,
            entry_row["timestamp"],
            candidate["timestamp"],
            entry_row["predictions"],
            candidate["predictions"],
            entry_actual_value,
            candidate_actual_value,
        )
    )

strat_df = pd.DataFrame(strategies)
st.subheader("Strategy comparison")
st.dataframe(strat_df)

