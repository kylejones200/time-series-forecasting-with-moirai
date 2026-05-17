"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def filter_only_use_1_rows_again() -> None:
    strict_use_df = arms_df[arms_df['use'] == 1].copy()

    strict_use_df = strict_use_df.sort_values(['ccode', 'techname', 'year'])

    strict_use_df['used_shift'] = strict_use_df.groupby(['ccode', 'techname'])['use'].shift(1, fill_value=0)

    strict_use_df['new_period'] = (strict_use_df['use'] == 1) & (strict_use_df['used_shift'] != 1)

    strict_use_df['usage_group'] = strict_use_df.groupby(['ccode', 'techname'])['new_period'].cumsum()

    tech_type_spells = strict_use_df.groupby(['ccode', 'techname', 'techtype', 'usage_group']).agg(start_year=('year', 'min'), end_year=('year', 'max')).reset_index()

    tech_type_spells['duration_years'] = tech_type_spells['end_year'] - tech_type_spells['start_year'] + 1

    active_2023 = strict_use_df[(strict_use_df['year'] == 2023) & (strict_use_df['use'] == 1)][['ccode', 'techname', 'usage_group']]

    active_2023['still_used_2023'] = True

    tech_type_spells = tech_type_spells.merge(active_2023, on=['ccode', 'techname', 'usage_group'], how='left')

    tech_type_spells['still_used_2023'] = tech_type_spells['still_used_2023'].fillna(False)

    tech_type_spells['event_observed'] = (~tech_type_spells['still_used_2023']).astype(int)

