"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def step_1_define_countries_and_time_window() -> None:
    nato_ccodes = [2, 20, 200, 220, 225, 255, 290, 310, 325, 365, 366, 368, 369, 370, 372, 385, 402, 404, 410, 420, 421, 423, 775, 840]

    named_nato = {2: 'USA', 20: 'Canada', 220: 'France', 200: 'UK'}

    spells['nato_group'] = spells['ccode'].apply(label_nato_group)

    nato_spells = spells[spells['nato_group'].notna() & (spells['start_year'] >= 1950)]

    plt.figure(figsize=(10, 6))

    kmf = KaplanMeierFitter()

    for group in ['USA', 'UK', 'Canada', 'France', 'Other NATO']:
        subset = nato_spells[nato_spells['nato_group'] == group]
        kmf.fit(subset['duration_years'], event_observed=subset['event_observed'], label=group)
        kmf.plot_survival_function(ci_show=False)

    plt.title('Military Technology Survival by NATO Member (1950–present, use == 1 only)')

    plt.xlabel('Years of Use')

    plt.ylabel('Survival Probability')

    plt.grid(False)

    plt.legend(title='NATO Member')

    plt.tight_layout()

    plt.show()

