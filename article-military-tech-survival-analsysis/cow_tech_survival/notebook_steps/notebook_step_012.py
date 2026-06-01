"""Notebook steps (auto-split)."""

import itertools

from lifelines.statistics import logrank_test


def notebook_step_012() -> None:
    periods = us_spells["us_period"].unique()
    for p1, p2 in itertools.combinations(periods, 2):
        df1 = us_spells[us_spells["us_period"] == p1]
        df2 = us_spells[us_spells["us_period"] == p2]
        result = logrank_test(
            df1["duration_years"],
            df2["duration_years"],
            event_observed_A=df1["event_observed"],
            event_observed_B=df2["event_observed"],
        )
        print(
            f"{p1} vs {p2} — p-value: {result.p_value:.4f}, chi²: {result.test_statistic:.2f}"
        )
