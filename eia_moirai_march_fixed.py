import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

np.random.seed(42)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    }
)


def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    context_len: int = 512
    horizon: int = 8
    model_id: str = "Salesforce/moirai-1.0-R-small"


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def main():
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

    predictor = model.create_predictor(batch_size=1, device="cpu")
    start_ts = y_train.index[0]
    dataset = ListDataset(
        [{"target": y_train.values.astype(np.float32), "start": start_ts}], freq="M"
    )

    it = predictor.predict(dataset)
    fc_mean = None
    for f in it:
        try:
            fc_mean = f.mean
        except Exception:
            # fallback to samples mean
            fc_mean = f.samples.mean(axis=0)
        break

    dates = pd.period_range("2025-01", "2025-08", freq="M").to_timestamp()
    fc = pd.Series(np.asarray(fc_mean).reshape(-1)[: cfg.horizon], index=dates)

    # Greyscale Tufte-style plot
    start_2024 = pd.Timestamp("2024-01-01")
    y_hist = y.loc[start_2024:end_2024]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_hist.index, y_hist.values, color="#888888", lw=1.5)
    ax.axvline(jan_2025, color="#666666", linestyle="--", lw=1)
    if len(y_act):
        ax.plot(y_act.index, y_act.values, color="#444444", lw=1.8)
    ax.plot(fc.index, fc.values, color="#000000", lw=2.0)

    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel("")

    if len(y_hist):
        ax.annotate(
            "History (2024)",
            xy=(y_hist.index[-1], y_hist.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#666666",
        )
    if len(y_act):
        ax.annotate(
            "Actual (Jan–Aug 2025)",
            xy=(y_act.index[-1], y_act.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="#444444",
        )
    ax.annotate(
        "Moirai",
        xy=(fc.index[-1], fc.values[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        va="center",
        ha="left",
        color="#000000",
    )

    ax.set_title("EIA Net Generation — Moirai forecast Jan–Aug 2025")
    save_fig("eia_moirai_last_fold.png")


if __name__ == "__main__":
    main()
