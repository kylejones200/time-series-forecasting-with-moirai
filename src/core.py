"""Core functions for Moirai time series forecasting."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule


def plot_single(
    inp: Any,
    label: Any,
    forecast: Any,
    *,
    context_length: int,
    name: str,
    show_label: bool = True,
) -> None:
    """Plot one forecast window (context + label + forecast)."""
    context = np.asarray(inp)[-context_length:]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(context)), context, label="context", color="#4A90A4")
    if show_label and label is not None:
        label_arr = np.asarray(label)
        offset = len(context)
        ax.plot(
            range(offset, offset + len(label_arr)),
            label_arr,
            label="label",
            color="#444444",
        )
    forecast_arr = np.asarray(getattr(forecast, "mean", forecast))
    offset = len(context) + (len(np.asarray(label)) if show_label and label is not None else 0)
    ax.plot(
        range(offset, offset + len(forecast_arr)),
        forecast_arr,
        label="forecast",
        color="#D4A574",
    )
    ax.set_title(name)
    ax.legend(loc="best")

def load_data(data_path: Path, date_column: str='Date', value_column: str='ERCOT', freq: str='h') -> pd.DataFrame:
    """Load and prepare time series data."""
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df[date_column])
    df.set_index('timestamp', inplace=True)
    df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
    df = df.asfreq(freq)
    return df

def create_dataset(df: pd.DataFrame, value_column: str) -> PandasDataset:
    """Convert DataFrame to GluonTS PandasDataset."""
    return PandasDataset(dict(df[[value_column]]))

def split_dataset(dataset: PandasDataset, test_size: int) -> tuple:
    """Split dataset into train and test."""
    train, test_template = split(dataset, offset=-test_size)
    test_data = test_template.generate_instances(prediction_length=test_size, windows=1, distance=test_size)
    return (train, test_data)

def create_moirai_model(dataset: PandasDataset, size: str='small', prediction_length: int=64, context_length: int=200, patch_size: str='auto', num_samples: int=100) -> MoiraiForecast:
    """Create Moirai forecasting model."""
    model = MoiraiForecast(module=MoiraiModule.from_pretrained(f'Salesforce/moirai-1.0-R-{size}'), prediction_length=prediction_length, context_length=context_length, patch_size=patch_size, num_samples=num_samples, target_dim=1, feat_dynamic_real_dim=dataset.num_feat_dynamic_real, past_feat_dynamic_real_dim=dataset.num_past_feat_dynamic_real)
    return model

def generate_forecasts(model: MoiraiForecast, test_data, batch_size: int=32):
    """Generate forecasts using Moirai model."""
    predictor = model.create_predictor(batch_size=batch_size)
    forecasts = predictor.predict(test_data.input)
    return forecasts

def save_forecast_plot(test_data, forecasts, context_length: int, output_path: Path):
    """Save forecast visualization."""
    input_it = iter(test_data.input)
    label_it = iter(test_data.label)
    forecast_it = iter(forecasts)
    inp = next(input_it)
    label = next(label_it)
    forecast = next(forecast_it)
    plot_single(inp, label, forecast, context_length=context_length, name='moirai_forecast', show_label=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
