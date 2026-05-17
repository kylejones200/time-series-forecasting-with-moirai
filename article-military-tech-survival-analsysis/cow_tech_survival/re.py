"""Auto-split from legacy monolithic script."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def re() -> None:
    # Re-import libraries after environment reset
    # import matplotlib.pyplot as plt
    import pandas as pd
    from lifelines import KaplanMeierFitter

    # Load previously uploaded file again (if needed)
    file_path = "cow_arms_tech_long.csv"
    arms_df = pd.read_csv(file_path)

    # Sort and preprocess
    arms_df = arms_df.sort_values(["ccode", "techname", "year"])
    arms_df["used"] = arms_df["total_use"].notna() & (arms_df["total_use"] > 0)
    arms_df["used_shift"] = arms_df.groupby(["ccode", "techname"])["used"].shift(
        1, fill_value=False
    )
    arms_df["new_period"] = arms_df["used"] & (~arms_df["used_shift"])
    arms_df["usage_group"] = arms_df.groupby(["ccode", "techname"])["new_period"].cumsum()
    arms_df.loc[~arms_df["used"], "usage_group"] = pd.NA

    # Aggregate usage spells
    spell_df = (
        arms_df.dropna(subset=["usage_group"])
        .groupby(["ccode", "techname", "usage_group"])
        .agg(start_year=("year", "min"), end_year=("year", "max"))
        .reset_index()
    )

    spell_df["duration_years"] = spell_df["end_year"] - spell_df["start_year"] + 1
    spell_df["event_observed"] = (spell_df["end_year"] != 2024).astype(int)


    # Label periods for US only (ccode == 2)
    def label_us_period(row):
        if row["ccode"] != 2:
            return None
        if row["start_year"] < 1916:
            return "1816–1915"
        elif row["start_year"] < 1941:
            return "1916–1940"
        elif row["start_year"] <= 1991:
            return "1945–1991"
        else:
            return "1992+"


    spell_df["us_period"] = spell_df.apply(label_us_period, axis=1)
    us_spells = spell_df[spell_df["us_period"].notna()].copy()

    # Plot Kaplan-Meier curves by historical period
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    for period in us_spells["us_period"].unique():
        subset = us_spells[us_spells["us_period"] == period]
        kmf.fit(
            subset["duration_years"], event_observed=subset["event_observed"], label=period
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title("U.S. Military Technology Survival by Historical Period")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.legend(title="Period")
    plt.tight_layout()
    plt.show()

