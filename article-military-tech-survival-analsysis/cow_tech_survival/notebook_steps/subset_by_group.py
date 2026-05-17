"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def subset_by_group() -> None:
    nato = grouped_spells[grouped_spells['group'] == 'NATO']

    china = grouped_spells[grouped_spells['group'] == 'China']

    russia = grouped_spells[grouped_spells['group'] == 'Russia/USSR']

    print('NATO vs China')

    print(logrank_test(nato['duration_years'], china['duration_years'], event_observed_A=nato['event_observed'], event_observed_B=china['event_observed']).summary)

    print('NATO vs Russia')

    print(logrank_test(nato['duration_years'], russia['duration_years'], event_observed_A=nato['event_observed'], event_observed_B=russia['event_observed']).summary)

    print('China vs Russia')

    print(logrank_test(china['duration_years'], russia['duration_years'], event_observed_A=china['event_observed'], event_observed_B=russia['event_observed']).summary)

