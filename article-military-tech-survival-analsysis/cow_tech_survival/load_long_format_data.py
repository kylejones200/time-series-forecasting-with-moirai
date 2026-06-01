"""Auto-split from legacy monolithic script."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def load_long_format_data() -> None:
    df = pd.read_csv("cow_arms_tech_long.csv")
    df = df[df["use"] == 1].copy()
    df = df.sort_values(["ccode", "techname", "year"])
    df["used_shift"] = df.groupby(["ccode", "techname"])["use"].shift(1, fill_value=0)
    df["new_spell"] = (df["use"] == 1) & (df["used_shift"] != 1)
    df["spell_id"] = df.groupby(["ccode", "techname"])["new_spell"].cumsum()
    spells = (
        df.groupby(["ccode", "techname", "techtype", "spell_id"])
        .agg(start_year=("year", "min"), end_year=("year", "max"))
        .reset_index()
    )
    spells["duration_years"] = spells["end_year"] - spells["start_year"] + 1
    spells["event_observed"] = (spells["end_year"] < 2023).astype(int)
    top_types = spells["techtype"].value_counts().head(5).index
    plt.figure(figsize=(12, 7))
    kmf = KaplanMeierFitter()
    for tech in top_types:
        subset = spells[spells["techtype"] == tech]
        kmf.fit(subset["duration_years"], subset["event_observed"], label=tech)
        kmf.plot_survival_function(ci_show=True)

    plt.title("Technology Survival by Type (use == 1 only)")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("km_techtype_top5.png")
    plt.close()
    results = []
    tech_groups = {
        t: df
        for t, df in spells[spells["techtype"].isin(top_types)].groupby("techtype")
    }
    for a, b in itertools.combinations(top_types, 2):
        df1, df2 = (tech_groups[a], tech_groups[b])
        result = logrank_test(
            df1["duration_years"],
            df2["duration_years"],
            event_observed_A=df1["event_observed"],
            event_observed_B=df2["event_observed"],
        )
        results.append(
            (a, b, result.test_statistic, result.p_value, -np.log2(result.p_value))
        )

    logrank_df = pd.DataFrame(
        results, columns=["TechType 1", "TechType 2", "Chi²", "p-value", "-log2(p)"]
    )
    logrank_df.to_csv("logrank_results_techtypes.csv", index=False)
    print("Log-rank test results saved to logrank_results_techtypes.csv")
