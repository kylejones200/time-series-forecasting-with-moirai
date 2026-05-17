"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_a_lookup_for_ccode_techname_usage_group_e() -> None:
    active_2023 = arms_df[(arms_df['year'] == 2023) & arms_df['used']][['ccode', 'techname', 'usage_group']]

    active_2023['still_used_2023'] = True

    spell_df = spell_df.merge(active_2023, on=['ccode', 'techname', 'usage_group'], how='left')

    spell_df['still_used_2023'] = spell_df['still_used_2023'].fillna(False)

    spell_df['event_observed'] = ~spell_df['still_used_2023']

    spell_df['event_observed'] = spell_df['event_observed'].astype(int)

