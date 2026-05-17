"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def label_rows_as_usa_or_other() -> None:
    tech_type_spells['group'] = tech_type_spells['ccode'].apply(lambda x: 'USA' if x == 2 else 'Other')

    plt.figure(figsize=(10, 6))

    kmf = KaplanMeierFitter()

    for group in ['USA', 'Other']:
        subset = tech_type_spells[tech_type_spells['group'] == group]
        kmf.fit(subset['duration_years'], event_observed=subset['event_observed'], label=group)
        kmf.plot_survival_function(ci_show=True)

    plt.title('Technology Survival: USA vs. All Other Countries (use == 1 only)')

    plt.xlabel('Years of Use')

    plt.ylabel('Survival Probability')

    plt.grid(True, linestyle='--', alpha=0.3)

    plt.legend(title='Group')

    plt.tight_layout()

    plt.show()

