"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main() -> None:
    notebook_step_002()
    initialize_the_kaplan_meier_fitter()
    start_by_identifying_nato_countries()
    split_groups()
    define_country_groups_by_ccode()
    get_the_groups_you_compared()
    notebook_step_009()
    notebook_step_010()
    notebook_step_011()
    notebook_step_012()
    re()
    load_your_dataset()
    how_many_spells_per_us_period()
    notebook_step_016()
    notebook_step_017()
    step_1_mark_technologies_as_used_based_on_correc()
    proceed_with_kaplan_meier_plot_comparing_u_s_his()
    create_a_lookup_for_ccode_techname_usage_group_e()
    recalculate_correct_censoring_logic()
    notebook_step_022()
    notebook_step_023()
    filter_the_arms_df_to_only_include_use_1()
    filter_spells_to_only_include_those_that_began_i()
    subset_by_group()
    step_1_define_countries_and_time_window()
    prepare_groups()
    define_the_50_year_bins()
    group_the_data_by_time_bin()
    filter_only_use_1_rows_again()
    plot_kaplan_meier_survival_curves_by_techtype_fo()
    custom_kaplan_meier_plot_with_formatting_improve()
    compare_top_5_most_common_tech_types_pairwise()
    notebook_step_035()
    label_rows_as_usa_or_other()
    notebook_step_037()
    load_long_format_data()
    load_long_format_data_2()
    filter_to_only_small_arms_usage_spells()
    prepare_the_data_for_cox_proportional_hazards_mo()
    notebook_step_042()
    filter_the_spells_to_only_small_arms_and_fighter()

