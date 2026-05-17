"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_kaplan_meier_survival_curves_by_techtype_fo() -> None:
    plt.figure(figsize=(12, 7))

    kmf = KaplanMeierFitter()

    top_techtypes = tech_type_spells['techtype'].value_counts().head(8).index

    for techtype in top_techtypes:
        subset = tech_type_spells[tech_type_spells['techtype'] == techtype]
        kmf.fit(subset['duration_years'], event_observed=subset['event_observed'], label=techtype)
        kmf.plot_survival_function(ci_show=False)

    plt.title('Technology Survival by Type (use == 1 only)')

    plt.xlabel('Years of Use')

    plt.ylabel('Survival Probability')

    plt.grid(True, linestyle='--', alpha=0.3)

    plt.legend(title='Technology Type')

    plt.tight_layout()

    plt.show()

