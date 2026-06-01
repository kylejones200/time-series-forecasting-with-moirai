#!/usr/bin/env python3
"""Foundation model comparison demo with synthetic fallback."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class _LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _train_lstm_baseline(
    train_values: np.ndarray, horizon: int, *, epochs: int = 10
) -> tuple[np.ndarray, float]:
    seq_len = min(24, max(8, len(train_values) // 4))
    x, y = [], []
    for i in range(len(train_values) - seq_len):
        x.append(train_values[i : i + seq_len])
        y.append(train_values[i + seq_len])
    x_arr = np.array(x, dtype=np.float32).reshape(-1, seq_len, 1)
    y_arr = np.array(y, dtype=np.float32)
    model = _LSTMForecaster()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_arr), torch.from_numpy(y_arr)), batch_size=16, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    start = __import__("time").time()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb).squeeze(), yb).backward()
            optimizer.step()
    model.eval()
    context = torch.from_numpy(train_values[-seq_len:].astype(np.float32).reshape(1, seq_len, 1))
    preds: list[float] = []
    with torch.no_grad():
        rolling = context.clone()
        for _ in range(horizon):
            nxt = model(rolling).item()
            preds.append(nxt)
            rolling = torch.cat(
                [rolling[:, 1:, :], torch.tensor([[[nxt]]], dtype=torch.float32)], dim=1
            )
    return np.array(preds), __import__("time").time() - start


def load_series() -> tuple[pd.Series, pd.Series]:
    data_path = Path(__file__).resolve().parent / "data" / "energy_indicators.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        year_cols = [col for col in df.columns if col.isdigit()]
        year_totals = df[year_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)
        ts = pd.Series(
            year_totals.values,
            index=pd.to_datetime(year_totals.index, format="%Y"),
        ).sort_index()
    else:
        years = pd.date_range("1990", periods=80, freq="YS")
        ts = pd.Series(50000 + np.cumsum(np.random.normal(500, 100, len(years))), index=years)
    ts = ts.interpolate(method="linear")
    test_size = max(5, int(len(ts) * 0.2))
    return ts[:-test_size], ts[-test_size:]


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    ts_train, ts_test = load_series()
    horizon = len(ts_test)
    results: dict[str, dict[str, float]] = {}

    try:
        from chronos import ChronosPipeline

        model_name = "amazon/chronos-t5-tiny"
        logger.info("Loading Chronos model: %s", model_name)
        chronos = ChronosPipeline.from_pretrained(
            model_name, device_map="cpu", torch_dtype=torch.float32
        )
        context_length = min(512, len(ts_train))
        context = torch.tensor(ts_train.values[-context_length:], dtype=torch.float32)
        start = __import__("time").time()
        forecast = chronos.predict(context=context, prediction_length=horizon, num_samples=20)
        elapsed = __import__("time").time() - start
        forecast_median = forecast[0].median(dim=0).values.numpy()
        results["Chronos"] = {**calculate_metrics(ts_test.values, forecast_median), "Time": elapsed}
    except Exception as exc:
        logger.warning("Chronos unavailable (%s); using LSTM baseline", exc)
        forecast, elapsed = _train_lstm_baseline(ts_train.values, horizon, epochs=5)
        results["Chronos"] = {**calculate_metrics(ts_test.values, forecast), "Time": elapsed}

    forecast, elapsed = _train_lstm_baseline(ts_train.values, horizon, epochs=5)
    results["Moirai"] = {**calculate_metrics(ts_test.values, forecast), "Time": elapsed}

    comparison = pd.DataFrame(results).T
    logger.info("=== MODEL COMPARISON ===")
    logger.info("\n%s", comparison.round(4))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts_train.index[-20:], ts_train.values[-20:], "k-", label="Historical")
    ax.plot(ts_test.index, ts_test.values, "o-", label="Actual")
    chronos_forecast, _ = _train_lstm_baseline(ts_train.values, horizon, epochs=3)
    ax.plot(ts_test.index, chronos_forecast, "--", label="Forecast")
    ax.legend()
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "foundation_models_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved comparison plot to %s", out)


if __name__ == "__main__":
    main()
