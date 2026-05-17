"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def filter_spells_to_only_include_those_that_began_i() -> None:
    modern_spells = grouped_spells[grouped_spells['start_year'] >= 1940].copy()

    plt.figure(figsize=(10, 6))

    kmf = KaplanMeierFitter()

    for group in ['NATO', 'China', 'Russia/USSR']:
        subset = modern_spells[modern_spells['group'] == group]
        kmf.fit(subset['duration_years'], event_observed=subset['event_observed'], label=group)
        kmf.plot_survival_function(ci_show=False)

    plt.title('Military Technology Survival Since 1940 (use == 1 only)')

    plt.xlabel('Years of Use')

    plt.ylabel('Survival Probability')

    plt.grid(False)

    plt.legend(title='Group')

    plt.tight_layout()

    plt.show()

