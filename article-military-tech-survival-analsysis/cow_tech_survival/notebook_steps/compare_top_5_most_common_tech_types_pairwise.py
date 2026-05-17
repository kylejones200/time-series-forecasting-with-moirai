"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def compare_top_5_most_common_tech_types_pairwise() -> None:
    top_types = tech_type_spells['techtype'].value_counts().head(5).index

    tech_groups = {t: df for t, df in tech_type_spells[tech_type_spells['techtype'].isin(top_types)].groupby('techtype')}

    results = []

    for a, b in itertools.combinations(top_types, 2):
        df1, df2 = (tech_groups[a], tech_groups[b])
        result = logrank_test(df1['duration_years'], df2['duration_years'], event_observed_A=df1['event_observed'], event_observed_B=df2['event_observed'])
        results.append((a, b, result.test_statistic, result.p_value, -np.log2(result.p_value)))

    logrank_df = pd.DataFrame(results, columns=['TechType 1', 'TechType 2', 'Chi2', 'p-value', '-log2(p)'])

