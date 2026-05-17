"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def split_groups() -> None:
    nato_group = df[df['nato_member'] == 1]

    non_nato_group = df[df['nato_member'] == 0]

    result = logrank_test(nato_group['duration_years'], non_nato_group['duration_years'], event_observed_A=nato_group['event_observed'], event_observed_B=non_nato_group['event_observed'])

    result.print_summary()

