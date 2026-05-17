"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def how_many_spells_per_us_period() -> None:
    # How many spells per US period?
    us_period_counts = spell_df.query("ccode == 2")["start_year"].value_counts()
    print(us_period_counts)

