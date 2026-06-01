"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter


def load_your_dataset() -> None:
    df = pd.read_csv("cow_arms_tech_long.csv")
    df = df.sort_values(["ccode", "techname", "year"])
    df["used"] = df["total_use"].notna() & (df["total_use"] > 0)
    df["used_shift"] = df.groupby(["ccode", "techname"])["used"].shift(
        1, fill_value=False
    )
    df["new_period"] = df["used"] & ~df["used_shift"]
    df["usage_group"] = df.groupby(["ccode", "techname"])["new_period"].cumsum()
    df.loc[~df["used"], "usage_group"] = pd.NA
    spell_df = (
        df.dropna(subset=["usage_group"])
        .groupby(["ccode", "techname", "usage_group"])
        .agg(start_year=("year", "min"), end_year=("year", "max"))
        .reset_index()
    )
    spell_df["duration_years"] = spell_df["end_year"] - spell_df["start_year"] + 1
    spell_df["event_observed"] = (spell_df["end_year"] != 2024).astype(int)
    spell_df["us_period"] = spell_df.apply(label_us_period, axis=1)
    us_spells = spell_df[spell_df["us_period"].notna()].copy()
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for period in sorted(us_spells["us_period"].unique()):
        subset = us_spells[us_spells["us_period"] == period]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=period,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title("U.S. Military Technology Survival by Historical Period")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.legend(title="Period")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
