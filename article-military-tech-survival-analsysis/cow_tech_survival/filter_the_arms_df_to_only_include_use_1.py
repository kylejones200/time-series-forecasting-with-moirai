"""Auto-split from legacy monolithic script."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def filter_the_arms_df_to_only_include_use_1() -> None:
    strict_use_df = arms_df[arms_df['use'] == 1].copy()

    strict_use_df = strict_use_df.sort_values(['ccode', 'techname', 'year'])

    strict_use_df['used_shift'] = strict_use_df.groupby(['ccode', 'techname'])['use'].shift(1, fill_value=0)

    strict_use_df['new_period'] = (strict_use_df['use'] == 1) & (strict_use_df['used_shift'] != 1)

    strict_use_df['usage_group'] = strict_use_df.groupby(['ccode', 'techname'])['new_period'].cumsum()

    spells = strict_use_df.groupby(['ccode', 'techname', 'usage_group']).agg(start_year=('year', 'min'), end_year=('year', 'max')).reset_index()

    spells['duration_years'] = spells['end_year'] - spells['start_year'] + 1

    active_2023 = strict_use_df[(strict_use_df['year'] == 2023) & (strict_use_df['use'] == 1)][['ccode', 'techname', 'usage_group']]

    active_2023['still_used_2023'] = True

    spells = spells.merge(active_2023, on=['ccode', 'techname', 'usage_group'], how='left')

    spells['still_used_2023'] = spells['still_used_2023'].fillna(False)

    spells['event_observed'] = (~spells['still_used_2023']).astype(int)

    nato_ccodes = [2, 20, 200, 220, 225, 255, 290, 310, 325, 365, 366, 368, 369, 370, 372, 385, 402, 404, 410, 420, 421, 423, 775, 840]

    china_ccode = [710]

    russia_ccodes = [365, 364]

    spells['group'] = spells['ccode'].apply(assign_group)

    grouped_spells = spells[spells['group'].notna()]

    plt.figure(figsize=(10, 6))

    kmf = KaplanMeierFitter()

    for group in ['NATO', 'China', 'Russia/USSR']:
        subset = grouped_spells[grouped_spells['group'] == group]
        kmf.fit(subset['duration_years'], event_observed=subset['event_observed'], label=group)
        kmf.plot_survival_function(ci_show=False)

    plt.title('Military Technology Survival: NATO, China, Russia/USSR (use == 1 only)')

    plt.xlabel('Years of Use')

    plt.ylabel('Survival Probability')

    plt.legend(title='Group')

    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    plt.show()

