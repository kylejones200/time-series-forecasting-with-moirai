"""Notebook steps (auto-split)."""

import itertools

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test


def get_the_groups_you_compared() -> None:
    groups = [
        "NATO",
        "Post-USSR Non-NATO",
        "China",
        "Africa",
        "Western Hemisphere",
        "Asia excl. China",
    ]
    logrank_results = []
    for g1, g2 in itertools.combinations(groups, 2):
        df1 = spell_df[spell_df["region_group"] == g1]
        df2 = spell_df[spell_df["region_group"] == g2]
        result = logrank_test(
            df1["duration_years"],
            df2["duration_years"],
            event_observed_A=df1["event_observed"],
            event_observed_B=df2["event_observed"],
        )
        logrank_results.append(
            {
                "Group 1": g1,
                "Group 2": g2,
                "Chi2": result.test_statistic,
                "p-value": result.p_value,
                "-log2(p)": -np.log2(result.p_value)
                if result.p_value > 0
                else float("inf"),
            }
        )

    logrank_df = pd.DataFrame(logrank_results).sort_values("p-value")
    print(logrank_df)
