"""Auto-split from legacy monolithic script."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def define_country_groups_by_ccode() -> None:
    spell_df = pd.read_csv('/content/Technology_Usage_Spells.csv')

    nato_ccodes = [2, 20, 200, 220, 225, 255, 290, 310, 325, 365, 366, 368, 369, 370, 372, 385, 402, 404, 410, 420, 421, 423, 775, 840]

    ussr_successor_non_nato = [365, 364, 369, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 385]

    ussr_successor_non_nato = [cc for cc in ussr_successor_non_nato if cc not in nato_ccodes]

    china_ccode = [710]

    africa_ccodes = list(range(500, 621))

    western_hemisphere_ccodes = [cc for cc in range(2, 399) if cc not in nato_ccodes]

    asia_ccodes = [cc for cc in range(630, 700) if cc not in china_ccode and cc not in nato_ccodes]

    spell_df['region_group'] = spell_df['ccode'].apply(assign_group)

    desired_groups = ['NATO', 'Post-USSR Non-NATO', 'China', 'Africa', 'Western Hemisphere', 'Asia excl. China']

    filtered_df = spell_df[spell_df['region_group'].isin(desired_groups)]

    plt.figure(figsize=(12, 7))

    kmf = KaplanMeierFitter()

    for group in desired_groups:
        subset = filtered_df[filtered_df['region_group'] == group]
        kmf.fit(durations=subset['duration_years'], event_observed=subset['event_observed'], label=group)
        kmf.plot_survival_function(ci_show=False)

    plt.title('Technology Usage Duration by Country Group')

    plt.xlabel('Years of Use')

    plt.ylabel('Survival Probability')

    plt.legend(title='Region Group')

    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    plt.show()

