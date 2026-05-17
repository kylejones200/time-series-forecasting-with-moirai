"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def notebook_step_017() -> None:
    us_subset = arms_df[arms_df['ccode'] == 2]

    us_subset.groupby('techname')['total_use'].nunique().sort_values()

