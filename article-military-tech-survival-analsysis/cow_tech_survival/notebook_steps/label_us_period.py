"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def label_us_period(row):
    if row['ccode'] != 2:
        return None
    if row['start_year'] < 1916:
        return '1816–1915'
    elif row['start_year'] < 1941:
        return '1916–1940'
    elif row['start_year'] <= 1991:
        return '1945–1991'
    else:
        return '1992+'

