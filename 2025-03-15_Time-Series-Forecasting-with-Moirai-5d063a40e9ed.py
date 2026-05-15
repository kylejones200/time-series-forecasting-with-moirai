# Description: Short example for Time Series Forecasting with Moirai.




from data_io import read_csv
from dataclasses import dataclass
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from pathlib import Path
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import signalplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SIZE = "small"  # Model size
PDT = 64  # Prediction length
CTX = 200  # Context length
PSZ = "auto"  # Patch size
BSZ = 32  # Batch size
TEST = 64  # Test set length
# Load data
df = read_csv("Ercot_Native_Load_2025 (1).csv")
df["timestamp"] = pd.to_datetime(df["Date"])
df.set_index("timestamp", inplace=True)
df["ERCOT"] = pd.to_numeric(df["ERCOT"], errors="coerce")
df = df.asfreq("h")  # Hourly frequency
# Convert into GluonTS dataset
ds = PandasDataset(dict(df[["ERCOT"]]))

train, test_template = split(ds, offset=-TEST)
test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST // PDT,
    distance=PDT,
)

model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{SIZE}"),
    prediction_length=PDT,
    context_length=CTX,
    patch_size=PSZ,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
)

predictor = model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data.input)

input_it = iter(test_data.input)
label_it = iter(test_data.label)
forecast_it = iter(forecasts)

inp = next(input_it)
label = next(label_it)
forecast = next(forecast_it)
plot_single(
    inp,
    label,
    forecast,
    context_length=CTX,
    name="moirai_forecast",
    show_label=True,
)
plt.show()


signalplot.apply(font_family='serif')


@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    context_len: int = 512
    horizon: int = 8
    model_id: str = "Salesforce/moirai-1.0-R-small"


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def main(plot: bool = False):
    np.random.seed(42)
    from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
    from gluonts.dataset.common import ListDataset

    cfg = Config()
    y = load_series(cfg)

    end_2024 = pd.Timestamp("2024-12-01")
    jan_2025 = pd.Timestamp("2025-01-01")
    aug_2025 = pd.Timestamp("2025-08-01")

    y_train = y.loc[:end_2024]
    y_act = y.loc[jan_2025:aug_2025]

    module = MoiraiModule.from_pretrained(cfg.model_id)
    model = MoiraiForecast(
        prediction_length=cfg.horizon,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=cfg.context_len,
        module=module,
        num_samples=100,
    )

    predictor = model.create_predictor(batch_size=1, device='cpu')
    start_ts = y_train.index[0]
    dataset = ListDataset([{"target": y_train.values.astype(np.float32), "start": start_ts}], freq="M")

    it = predictor.predict(dataset)
    fc_mean = None
    for f in it:
        try:
            fc_mean = f.mean
        except Exception:
            # fallback to samples mean
            fc_mean = f.samples.mean(axis=0)
        break

    dates = pd.period_range('2025-01', '2025-08', freq='M').to_timestamp()
    fc = pd.Series(np.asarray(fc_mean).reshape(-1)[:cfg.horizon], index=dates)

    # Greyscale Tufte-style plot
    start_2024 = pd.Timestamp("2024-01-01")
    y_hist = y.loc[start_2024:end_2024]

    if plot:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(y_hist.index, y_hist.values, color="#888888", lw=1.5)
        ax.axvline(jan_2025, color="#666666", linestyle="--", lw=1)
        if len(y_act):
            ax.plot(y_act.index, y_act.values, color="#444444", lw=1.8)
        ax.plot(fc.index, fc.values, color="#000000", lw=2.0)

        from matplotlib.ticker import MaxNLocator, StrMethodFormatter
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('')

        if len(y_hist):
            ax.annotate('History (2024)', xy=(y_hist.index[-1], y_hist.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#666666')
        if len(y_act):
            ax.annotate('Actual (Jan-Aug 2025)', xy=(y_act.index[-1], y_act.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#444444')
        ax.annotate('Moirai', xy=(fc.index[-1], fc.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#000000')

        ax.set_title('EIA Net Generation — Moirai forecast Jan-Aug 2025')
        signalplot.save('eia_moirai_last_fold.png')

if __name__ == '__main__':
    main()
