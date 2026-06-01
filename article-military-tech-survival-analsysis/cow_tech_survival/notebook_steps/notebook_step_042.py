"""Notebook steps (auto-split)."""

import pandas as pd
from lifelines import CoxPHFitter


def notebook_step_042() -> None:
    cox_df = spells[["duration_years", "event_observed", "techtype"]].dropna().copy()
    cox_df = pd.get_dummies(cox_df, columns=["techtype"], drop_first=True)
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration_years", event_col="event_observed")
    cph.print_summary()
