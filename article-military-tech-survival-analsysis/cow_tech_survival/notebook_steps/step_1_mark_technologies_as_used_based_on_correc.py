"""Notebook steps (auto-split)."""

import pandas as pd


def step_1_mark_technologies_as_used_based_on_correc() -> None:
    arms_df = pd.read_csv("cow_arms_tech_long.csv")
    arms_df = arms_df.sort_values(["ccode", "techname", "year"])
    arms_df["used"] = arms_df["use"].isin([1])
    arms_df["used_shift"] = arms_df.groupby(["ccode", "techname"])["used"].shift(
        1, fill_value=False
    )
    arms_df["new_period"] = arms_df["used"] & ~arms_df["used_shift"]
    arms_df["usage_group"] = arms_df.groupby(["ccode", "techname"])[
        "new_period"
    ].cumsum()
    arms_df.loc[~arms_df["used"], "usage_group"] = pd.NA
    spell_df = (
        arms_df.dropna(subset=["usage_group"])
        .groupby(["ccode", "techname", "usage_group"])
        .agg(start_year=("year", "min"), end_year=("year", "max"))
        .reset_index()
    )
    spell_df["duration_years"] = spell_df["end_year"] - spell_df["start_year"] + 1
    spell_df["event_observed"] = (spell_df["end_year"] != 2023).astype(int)
    spell_df["us_period"] = spell_df.apply(label_us_period, axis=1)
    spell_df[spell_df["us_period"].notna()].copy()
