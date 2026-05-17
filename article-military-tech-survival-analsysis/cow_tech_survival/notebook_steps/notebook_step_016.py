"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def notebook_step_016() -> None:
    arms_df['used'] = arms_df['total_use'].notna() & (arms_df['total_use'] > 0)

