"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def start_by_identifying_nato_countries() -> None:
    nato_ccodes = [2, 20, 200, 220, 225, 255, 290, 310, 325, 365, 366, 368, 369, 370, 372, 385, 402, 404, 410, 420, 421, 423, 775, 840]

    df['nato_member'] = df['ccode'].isin(nato_ccodes).astype(int)

    kmf = KaplanMeierFitter()

    plt.figure(figsize=(10, 6))

    for group_value, group_name in zip([1, 0], ['NATO', 'Non-NATO']):
        subset = df[df['nato_member'] == group_value]
        kmf.fit(subset['duration_years'], subset['event_observed'], label=group_name)
        kmf.plot_survival_function(ci_show=False)

    plt.title('Technology Usage Duration: NATO vs. Non-NATO Countries')

    plt.xlabel('Years of Use')

    plt.ylabel('Survival Probability')

    plt.legend()

    plt.tight_layout()

    plt.show()

