"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def custom_kaplan_meier_plot_with_formatting_improve() -> None:
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

    plt.xlim(0, 50)

    plt.ylim(0.2, 1.0)

    plt.yticks([0.25, 0.5, 0.75])

    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)

    ax = plt.gca()

    ax.spines['right'].set_visible(False)

    ax.spines['top'].set_visible(False)

    plt.legend(title='Technology Type')

    plt.tight_layout()

    plt.show()

