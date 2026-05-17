"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def group_the_data_by_time_bin() -> None:
    import itertools

    import numpy as np
    from lifelines.statistics import logrank_test

    # Group the data by time bin
    us_groups = {period: group for period, group in us_binned_spells.groupby("us_bin")}

    # Generate all pairwise comparisons
    comparisons = list(itertools.combinations(us_groups.keys(), 2))

    # Run and print results
    for p1, p2 in comparisons:
        df1 = us_groups[p1]
        df2 = us_groups[p2]
        result = logrank_test(
            df1["duration_years"],
            df2["duration_years"],
            event_observed_A=df1["event_observed"],
            event_observed_B=df2["event_observed"],
        )
        print(f"{p1} vs {p2}")
        print(f"Chi² = {result.test_statistic:.4f}, p = {result.p_value:.4f}, -log2(p) = {-np.log2(result.p_value):.2f}\n"
        )

