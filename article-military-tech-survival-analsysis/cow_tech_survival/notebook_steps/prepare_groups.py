"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def prepare_groups() -> None:
    from lifelines.statistics import logrank_test

    # Prepare groups
    usa = nato_spells[nato_spells["nato_group"] == "USA"]
    uk = nato_spells[nato_spells["nato_group"] == "UK"]
    canada = nato_spells[nato_spells["nato_group"] == "Canada"]
    france = nato_spells[nato_spells["nato_group"] == "France"]
    other_nato = nato_spells[nato_spells["nato_group"] == "Other NATO"]

    # Define test pairs
    comparisons = {
        "USA vs UK": (usa, uk),
        "USA vs Canada": (usa, canada),
        "USA vs France": (usa, france),
        "USA vs Other NATO": (usa, other_nato),
    }

    # Run tests
    for name, (a, b) in comparisons.items():
        result = logrank_test(
            a["duration_years"],
            b["duration_years"],
            event_observed_A=a["event_observed"],
            event_observed_B=b["event_observed"],
        )
        print(f"{name}")
        print(
            f"Chi² = {result.test_statistic:.4f}, p = {result.p_value:.4f}, "
            f"-log2(p) = {-np.log2(result.p_value):.2f}\n"
        )

