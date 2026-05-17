"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def initialize_the_kaplan_meier_fitter() -> None:
    kmf = KaplanMeierFitter()

    kmf.fit(durations=df['duration_years'], event_observed=df['event_observed'])

    plt.figure(figsize=(10, 6))

    kmf.plot_survival_function()

    plt.title('Kaplan-Meier Estimate of Technology Usage Duration')

    plt.xlabel('Years of Use')

    plt.ylabel('Probability Technology Is Still in Use')

    plt.grid(False)

    plt.tight_layout()

    plt.show()

