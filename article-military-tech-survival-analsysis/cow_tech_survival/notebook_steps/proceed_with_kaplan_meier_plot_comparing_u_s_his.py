"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def proceed_with_kaplan_meier_plot_comparing_u_s_his() -> None:
    plt.figure(figsize=(10, 6))

    kmf = KaplanMeierFitter()

    for period in sorted(us_spells['us_period'].unique()):
        subset = us_spells[us_spells['us_period'] == period]
        kmf.fit(subset['duration_years'], event_observed=subset['event_observed'], label=period)
        kmf.plot_survival_function(ci_show=False)

    plt.title('U.S. Military Technology Survival by Historical Period')

    plt.xlabel('Years of Use')

    plt.ylabel('Survival Probability')

    plt.legend(title='Period')

    plt.grid(False)

    plt.tight_layout()

    plt.show()

