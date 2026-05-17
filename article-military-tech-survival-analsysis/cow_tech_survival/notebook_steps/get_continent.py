"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_continent(ccode):
    if ccode in africa_codes:
        return 'Africa'
    elif ccode in asia_codes:
        return 'Asia'
    elif ccode in europe_codes:
        return 'Europe'
    elif ccode in americas_codes:
        return 'Americas'
    elif ccode in middle_east_codes:
        return 'Middle East'
    else:
        return 'Other'

