"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def prepare_the_data_for_cox_proportional_hazards_mo() -> None:
    cox_df = spells[['duration_years', 'event_observed', 'techname']].dropna().copy()

    cox_df = pd.get_dummies(cox_df, columns=['techname'], drop_first=True)

    cph = CoxPHFitter()

    cph.fit(cox_df, duration_col='duration_years', event_col='event_observed')

    cph.print_summary()

