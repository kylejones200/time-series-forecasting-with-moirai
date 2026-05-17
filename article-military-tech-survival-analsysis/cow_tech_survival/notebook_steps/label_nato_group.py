"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def label_nato_group(ccode):
    if ccode == 2:
        return 'USA'
    elif ccode == 20:
        return 'Canada'
    elif ccode == 220:
        return 'France'
    elif ccode == 200:
        return 'UK'
    elif ccode in nato_ccodes:
        return 'Other NATO'
    else:
        return None

