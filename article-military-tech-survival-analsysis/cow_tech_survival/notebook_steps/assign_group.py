"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def assign_group(ccode):
    if ccode in china_ccode:
        return 'China'
    elif ccode in russia_ccodes:
        return 'Russia/USSR'
    elif ccode in nato_ccodes:
        return 'NATO'
    else:
        return None

