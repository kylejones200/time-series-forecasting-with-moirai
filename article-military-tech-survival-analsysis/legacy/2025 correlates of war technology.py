"""Generated from Jupyter notebook: 2025 correlates of war technology

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


def assign_group(ccode):
    if ccode in china_ccode:
        return "China"
    elif ccode in russia_ccodes:
        return "Russia/USSR"
    elif ccode in nato_ccodes:
        return "NATO"
    else:
        return None


def get_continent(ccode):
    if ccode in africa_codes:
        return "Africa"
    elif ccode in asia_codes:
        return "Asia"
    elif ccode in europe_codes:
        return "Europe"
    elif ccode in americas_codes:
        return "Americas"
    elif ccode in middle_east_codes:
        return "Middle East"
    else:
        return "Other"


def label_nato_group(ccode):
    if ccode == 2:
        return "USA"
    elif ccode == 20:
        return "Canada"
    elif ccode == 220:
        return "France"
    elif ccode == 200:
        return "UK"
    elif ccode in nato_ccodes:
        return "Other NATO"
    else:
        return None


def label_us_period(row):
    if row["ccode"] != 2:
        return None
    if row["start_year"] < 1916:
        return "1816–1915"
    elif row["start_year"] < 1941:
        return "1916–1940"
    elif row["start_year"] <= 1991:
        return "1945–1991"
    else:
        return "1992+"


def label_us_time_bin(row):
    if row["ccode"] != 2:
        return None
    elif 1923 <= row["start_year"] < 1973:
        return "1923–1973"
    elif 1973 <= row["start_year"] <= 2023:
        return "1973–2023"
    else:
        return None


def notebook_step_002() -> None:
    pd.read_csv("/content/Technology_Usage_Spells.csv")


def initialize_the_kaplan_meier_fitter() -> None:
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["duration_years"], event_observed=df["event_observed"])
    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function()
    plt.title("Kaplan-Meier Estimate of Technology Usage Duration")
    plt.xlabel("Years of Use")
    plt.ylabel("Probability Technology Is Still in Use")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def start_by_identifying_nato_countries() -> None:
    nato_ccodes = [
        2,
        20,
        200,
        220,
        225,
        255,
        290,
        310,
        325,
        365,
        366,
        368,
        369,
        370,
        372,
        385,
        402,
        404,
        410,
        420,
        421,
        423,
        775,
        840,
    ]
    df["nato_member"] = df["ccode"].isin(nato_ccodes).astype(int)
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    for group_value, group_name in zip([1, 0], ["NATO", "Non-NATO"]):
        subset = df[df["nato_member"] == group_value]
        kmf.fit(subset["duration_years"], subset["event_observed"], label=group_name)
        kmf.plot_survival_function(ci_show=False)

    plt.title("Technology Usage Duration: NATO vs. Non-NATO Countries")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


def split_groups() -> None:
    nato_group = df[df["nato_member"] == 1]
    non_nato_group = df[df["nato_member"] == 0]
    result = logrank_test(
        nato_group["duration_years"],
        non_nato_group["duration_years"],
        event_observed_A=nato_group["event_observed"],
        event_observed_B=non_nato_group["event_observed"],
    )
    result.print_summary()


def define_country_groups_by_ccode() -> None:
    spell_df = pd.read_csv("/content/Technology_Usage_Spells.csv")
    nato_ccodes = [
        2,
        20,
        200,
        220,
        225,
        255,
        290,
        310,
        325,
        365,
        366,
        368,
        369,
        370,
        372,
        385,
        402,
        404,
        410,
        420,
        421,
        423,
        775,
        840,
    ]
    ussr_successor_non_nato = [
        365,
        364,
        369,
        369,
        370,
        371,
        372,
        373,
        374,
        375,
        376,
        377,
        378,
        379,
        380,
        385,
    ]
    ussr_successor_non_nato = [
        cc for cc in ussr_successor_non_nato if cc not in nato_ccodes
    ]
    china_ccode = [710]
    list(range(500, 621))
    [cc for cc in range(2, 399) if cc not in nato_ccodes]
    [cc for cc in range(630, 700) if cc not in china_ccode and cc not in nato_ccodes]
    spell_df["region_group"] = spell_df["ccode"].apply(assign_group)
    desired_groups = [
        "NATO",
        "Post-USSR Non-NATO",
        "China",
        "Africa",
        "Western Hemisphere",
        "Asia excl. China",
    ]
    filtered_df = spell_df[spell_df["region_group"].isin(desired_groups)]
    plt.figure(figsize=(12, 7))
    kmf = KaplanMeierFitter()
    for group in desired_groups:
        subset = filtered_df[filtered_df["region_group"] == group]
        kmf.fit(
            durations=subset["duration_years"],
            event_observed=subset["event_observed"],
            label=group,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title("Technology Usage Duration by Country Group")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.legend(title="Region Group")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_the_groups_you_compared() -> None:
    groups = [
        "NATO",
        "Post-USSR Non-NATO",
        "China",
        "Africa",
        "Western Hemisphere",
        "Asia excl. China",
    ]
    logrank_results = []
    for g1, g2 in itertools.combinations(groups, 2):
        df1 = spell_df[spell_df["region_group"] == g1]
        df2 = spell_df[spell_df["region_group"] == g2]
        result = logrank_test(
            df1["duration_years"],
            df2["duration_years"],
            event_observed_A=df1["event_observed"],
            event_observed_B=df2["event_observed"],
        )
        logrank_results.append(
            {
                "Group 1": g1,
                "Group 2": g2,
                "Chi2": result.test_statistic,
                "p-value": result.p_value,
                "-log2(p)": -np.log2(result.p_value)
                if result.p_value > 0
                else float("inf"),
            }
        )

    logrank_df = pd.DataFrame(logrank_results).sort_values("p-value")
    print(logrank_df)


def notebook_step_009() -> None:
    spell_df["us_period"] = spell_df.apply(label_us_period, axis=1)


def notebook_step_010() -> None:
    spell_df[spell_df["us_period"].notna()].copy()


def notebook_step_011() -> None:
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for period in us_spells["us_period"].unique():
        subset = us_spells[us_spells["us_period"] == period]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=period,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title("U.S. Military Technology Survival by Historical Period")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.legend(title="Period")
    plt.tight_layout()
    plt.show()


def notebook_step_012() -> None:
    periods = us_spells["us_period"].unique()
    for p1, p2 in itertools.combinations(periods, 2):
        df1 = us_spells[us_spells["us_period"] == p1]
        df2 = us_spells[us_spells["us_period"] == p2]
        result = logrank_test(
            df1["duration_years"],
            df2["duration_years"],
            event_observed_A=df1["event_observed"],
            event_observed_B=df2["event_observed"],
        )
        print(
            f"{p1} vs {p2} — p-value: {result.p_value:.4f}, chi²: {result.test_statistic:.2f}"
        )


def re() -> None:
    # Re-import libraries after environment reset
    # import matplotlib.pyplot as plt
    import pandas as pd
    from lifelines import KaplanMeierFitter

    # Load previously uploaded file again (if needed)
    file_path = "cow_arms_tech_long.csv"
    arms_df = pd.read_csv(file_path)
    # Sort and preprocess
    arms_df = arms_df.sort_values(["ccode", "techname", "year"])
    arms_df["used"] = arms_df["total_use"].notna() & (arms_df["total_use"] > 0)
    arms_df["used_shift"] = arms_df.groupby(["ccode", "techname"])["used"].shift(
        1, fill_value=False
    )
    arms_df["new_period"] = arms_df["used"] & (~arms_df["used_shift"])
    arms_df["usage_group"] = arms_df.groupby(["ccode", "techname"])[
        "new_period"
    ].cumsum()
    arms_df.loc[~arms_df["used"], "usage_group"] = pd.NA
    # Aggregate usage spells
    spell_df = (
        arms_df.dropna(subset=["usage_group"])
        .groupby(["ccode", "techname", "usage_group"])
        .agg(start_year=("year", "min"), end_year=("year", "max"))
        .reset_index()
    )
    spell_df["duration_years"] = spell_df["end_year"] - spell_df["start_year"] + 1
    spell_df["event_observed"] = (spell_df["end_year"] != 2024).astype(int)

    # Label periods for US only (ccode == 2)
    def label_us_period(row):
        if row["ccode"] != 2:
            return None
        if row["start_year"] < 1916:
            return "1816–1915"
        elif row["start_year"] < 1941:
            return "1916–1940"
        elif row["start_year"] <= 1991:
            return "1945–1991"
        else:
            return "1992+"

    spell_df["us_period"] = spell_df.apply(label_us_period, axis=1)
    us_spells = spell_df[spell_df["us_period"].notna()].copy()
    # Plot Kaplan-Meier curves by historical period
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for period in us_spells["us_period"].unique():
        subset = us_spells[us_spells["us_period"] == period]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=period,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title("U.S. Military Technology Survival by Historical Period")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.legend(title="Period")
    plt.tight_layout()
    plt.show()


def load_your_dataset() -> None:
    df = pd.read_csv("cow_arms_tech_long.csv")
    df = df.sort_values(["ccode", "techname", "year"])
    df["used"] = df["total_use"].notna() & (df["total_use"] > 0)
    df["used_shift"] = df.groupby(["ccode", "techname"])["used"].shift(
        1, fill_value=False
    )
    df["new_period"] = df["used"] & ~df["used_shift"]
    df["usage_group"] = df.groupby(["ccode", "techname"])["new_period"].cumsum()
    df.loc[~df["used"], "usage_group"] = pd.NA
    spell_df = (
        df.dropna(subset=["usage_group"])
        .groupby(["ccode", "techname", "usage_group"])
        .agg(start_year=("year", "min"), end_year=("year", "max"))
        .reset_index()
    )
    spell_df["duration_years"] = spell_df["end_year"] - spell_df["start_year"] + 1
    spell_df["event_observed"] = (spell_df["end_year"] != 2024).astype(int)
    spell_df["us_period"] = spell_df.apply(label_us_period, axis=1)
    us_spells = spell_df[spell_df["us_period"].notna()].copy()
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for period in sorted(us_spells["us_period"].unique()):
        subset = us_spells[us_spells["us_period"] == period]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=period,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title("U.S. Military Technology Survival by Historical Period")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.legend(title="Period")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def how_many_spells_per_us_period() -> None:
    # How many spells per US period?
    us_period_counts = spell_df.query("ccode == 2")["start_year"].value_counts()
    print(us_period_counts)


def notebook_step_016() -> None:
    arms_df["used"] = arms_df["total_use"].notna() & (arms_df["total_use"] > 0)


def notebook_step_017() -> None:
    us_subset = arms_df[arms_df["ccode"] == 2]
    us_subset.groupby("techname")["total_use"].nunique().sort_values()


def step_1_mark_technologies_as_used_based_on_correc() -> None:
    arms_df = pd.read_csv("cow_arms_tech_long.csv")
    arms_df = arms_df.sort_values(["ccode", "techname", "year"])
    arms_df["used"] = arms_df["use"].isin([1])
    arms_df["used_shift"] = arms_df.groupby(["ccode", "techname"])["used"].shift(
        1, fill_value=False
    )
    arms_df["new_period"] = arms_df["used"] & ~arms_df["used_shift"]
    arms_df["usage_group"] = arms_df.groupby(["ccode", "techname"])[
        "new_period"
    ].cumsum()
    arms_df.loc[~arms_df["used"], "usage_group"] = pd.NA
    spell_df = (
        arms_df.dropna(subset=["usage_group"])
        .groupby(["ccode", "techname", "usage_group"])
        .agg(start_year=("year", "min"), end_year=("year", "max"))
        .reset_index()
    )
    spell_df["duration_years"] = spell_df["end_year"] - spell_df["start_year"] + 1
    spell_df["event_observed"] = (spell_df["end_year"] != 2023).astype(int)
    spell_df["us_period"] = spell_df.apply(label_us_period, axis=1)
    spell_df[spell_df["us_period"].notna()].copy()


def proceed_with_kaplan_meier_plot_comparing_u_s_his() -> None:
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for period in sorted(us_spells["us_period"].unique()):
        subset = us_spells[us_spells["us_period"] == period]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=period,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title("U.S. Military Technology Survival by Historical Period")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.legend(title="Period")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def create_a_lookup_for_ccode_techname_usage_group_e() -> None:
    active_2023 = arms_df[(arms_df["year"] == 2023) & arms_df["used"]][
        ["ccode", "techname", "usage_group"]
    ]
    active_2023["still_used_2023"] = True
    spell_df = spell_df.merge(
        active_2023, on=["ccode", "techname", "usage_group"], how="left"
    )
    spell_df["still_used_2023"] = spell_df["still_used_2023"].fillna(False)
    spell_df["event_observed"] = ~spell_df["still_used_2023"]
    spell_df["event_observed"] = spell_df["event_observed"].astype(int)


def recalculate_correct_censoring_logic() -> None:
    active_2023 = arms_df[(arms_df["year"] == 2023) & arms_df["used"]][
        ["ccode", "techname", "usage_group"]
    ]
    active_2023["still_used_2023"] = True
    spell_df = spell_df.merge(
        active_2023, on=["ccode", "techname", "usage_group"], how="left"
    )
    spell_df["still_used_2023"] = spell_df["still_used_2023"].fillna(False)
    spell_df["event_observed"] = (~spell_df["still_used_2023"]).astype(int)
    us_spells = spell_df[spell_df["us_period"].notna()].copy()
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for period in sorted(us_spells["us_period"].unique()):
        subset = us_spells[us_spells["us_period"] == period]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=period,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title("U.S. Military Technology Survival by Historical Period")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.legend(title="Period")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def notebook_step_022() -> None:
    us_spells.groupby("us_period")["event_observed"].value_counts()


def notebook_step_023() -> None:
    df.head()


def filter_the_arms_df_to_only_include_use_1() -> None:
    strict_use_df = arms_df[arms_df["use"] == 1].copy()
    strict_use_df = strict_use_df.sort_values(["ccode", "techname", "year"])
    strict_use_df["used_shift"] = strict_use_df.groupby(["ccode", "techname"])[
        "use"
    ].shift(1, fill_value=0)
    strict_use_df["new_period"] = (strict_use_df["use"] == 1) & (
        strict_use_df["used_shift"] != 1
    )
    strict_use_df["usage_group"] = strict_use_df.groupby(["ccode", "techname"])[
        "new_period"
    ].cumsum()
    spells = (
        strict_use_df.groupby(["ccode", "techname", "usage_group"])
        .agg(start_year=("year", "min"), end_year=("year", "max"))
        .reset_index()
    )
    spells["duration_years"] = spells["end_year"] - spells["start_year"] + 1
    active_2023 = strict_use_df[
        (strict_use_df["year"] == 2023) & (strict_use_df["use"] == 1)
    ][["ccode", "techname", "usage_group"]]
    active_2023["still_used_2023"] = True
    spells = spells.merge(
        active_2023, on=["ccode", "techname", "usage_group"], how="left"
    )
    spells["still_used_2023"] = spells["still_used_2023"].fillna(False)
    spells["event_observed"] = (~spells["still_used_2023"]).astype(int)
    spells["group"] = spells["ccode"].apply(assign_group)
    grouped_spells = spells[spells["group"].notna()]
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for group in ["NATO", "China", "Russia/USSR"]:
        subset = grouped_spells[grouped_spells["group"] == group]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=group,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title("Military Technology Survival: NATO, China, Russia/USSR (use == 1 only)")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.legend(title="Group")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def filter_spells_to_only_include_those_that_began_i() -> None:
    modern_spells = grouped_spells[grouped_spells["start_year"] >= 1940].copy()
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for group in ["NATO", "China", "Russia/USSR"]:
        subset = modern_spells[modern_spells["group"] == group]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=group,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title("Military Technology Survival Since 1940 (use == 1 only)")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.grid(False)
    plt.legend(title="Group")
    plt.tight_layout()
    plt.show()


def subset_by_group() -> None:
    nato = grouped_spells[grouped_spells["group"] == "NATO"]
    china = grouped_spells[grouped_spells["group"] == "China"]
    russia = grouped_spells[grouped_spells["group"] == "Russia/USSR"]
    print("NATO vs China")
    print(
        logrank_test(
            nato["duration_years"],
            china["duration_years"],
            event_observed_A=nato["event_observed"],
            event_observed_B=china["event_observed"],
        ).summary
    )
    print("NATO vs Russia")
    print(
        logrank_test(
            nato["duration_years"],
            russia["duration_years"],
            event_observed_A=nato["event_observed"],
            event_observed_B=russia["event_observed"],
        ).summary
    )
    print("China vs Russia")
    print(
        logrank_test(
            china["duration_years"],
            russia["duration_years"],
            event_observed_A=china["event_observed"],
            event_observed_B=russia["event_observed"],
        ).summary
    )


def step_1_define_countries_and_time_window() -> None:
    spells["nato_group"] = spells["ccode"].apply(label_nato_group)
    nato_spells = spells[spells["nato_group"].notna() & (spells["start_year"] >= 1950)]
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for group in ["USA", "UK", "Canada", "France", "Other NATO"]:
        subset = nato_spells[nato_spells["nato_group"] == group]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=group,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title(
        "Military Technology Survival by NATO Member (1950–present, use == 1 only)"
    )
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.grid(False)
    plt.legend(title="NATO Member")
    plt.tight_layout()
    plt.show()


def prepare_groups() -> None:
    from lifelines.statistics import logrank_test

    # Prepare groups
    usa = nato_spells[nato_spells["nato_group"] == "USA"]
    uk = nato_spells[nato_spells["nato_group"] == "UK"]
    canada = nato_spells[nato_spells["nato_group"] == "Canada"]
    france = nato_spells[nato_spells["nato_group"] == "France"]
    other_nato = nato_spells[nato_spells["nato_group"] == "Other NATO"]
    # Define test pairs
    comparisons = {
        "USA vs UK": (usa, uk),
        "USA vs Canada": (usa, canada),
        "USA vs France": (usa, france),
        "USA vs Other NATO": (usa, other_nato),
    }
    # Run tests
    for name, (a, b) in comparisons.items():
        result = logrank_test(
            a["duration_years"],
            b["duration_years"],
            event_observed_A=a["event_observed"],
            event_observed_B=b["event_observed"],
        )
        print(f"{name}")
        print(
            f"Chi² = {result.test_statistic:.4f}, p = {result.p_value:.4f}, "
            f"-log2(p) = {-np.log2(result.p_value):.2f}\n"
        )


def define_the_50_year_bins() -> None:
    spells["us_bin"] = spells.apply(label_us_time_bin, axis=1)
    us_binned_spells = spells[spells["us_bin"].notna()].copy()
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for bin_label in sorted(us_binned_spells["us_bin"].unique()):
        subset = us_binned_spells[us_binned_spells["us_bin"] == bin_label]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=bin_label,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title("U.S. Military Technology Survival by 50-Year Period (use == 1 only)")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.grid(False)
    plt.legend(title="Time Period")
    plt.tight_layout()
    plt.show()


def group_the_data_by_time_bin() -> None:
    import itertools

    import numpy as np
    from lifelines.statistics import logrank_test

    # Group the data by time bin
    us_groups = {period: group for period, group in us_binned_spells.groupby("us_bin")}
    # Generate all pairwise comparisons
    comparisons = list(itertools.combinations(us_groups.keys(), 2))
    # Run and print results
    for p1, p2 in comparisons:
        df1 = us_groups[p1]
        df2 = us_groups[p2]
        result = logrank_test(
            df1["duration_years"],
            df2["duration_years"],
            event_observed_A=df1["event_observed"],
            event_observed_B=df2["event_observed"],
        )
        print(f"{p1} vs {p2}")
        print(
            f"Chi² = {result.test_statistic:.4f}, p = {result.p_value:.4f}, -log2(p) = {-np.log2(result.p_value):.2f}\n"
        )


def filter_only_use_1_rows_again() -> None:
    strict_use_df = arms_df[arms_df["use"] == 1].copy()
    strict_use_df = strict_use_df.sort_values(["ccode", "techname", "year"])
    strict_use_df["used_shift"] = strict_use_df.groupby(["ccode", "techname"])[
        "use"
    ].shift(1, fill_value=0)
    strict_use_df["new_period"] = (strict_use_df["use"] == 1) & (
        strict_use_df["used_shift"] != 1
    )
    strict_use_df["usage_group"] = strict_use_df.groupby(["ccode", "techname"])[
        "new_period"
    ].cumsum()
    tech_type_spells = (
        strict_use_df.groupby(["ccode", "techname", "techtype", "usage_group"])
        .agg(start_year=("year", "min"), end_year=("year", "max"))
        .reset_index()
    )
    tech_type_spells["duration_years"] = (
        tech_type_spells["end_year"] - tech_type_spells["start_year"] + 1
    )
    active_2023 = strict_use_df[
        (strict_use_df["year"] == 2023) & (strict_use_df["use"] == 1)
    ][["ccode", "techname", "usage_group"]]
    active_2023["still_used_2023"] = True
    tech_type_spells = tech_type_spells.merge(
        active_2023, on=["ccode", "techname", "usage_group"], how="left"
    )
    tech_type_spells["still_used_2023"] = tech_type_spells["still_used_2023"].fillna(
        False
    )
    tech_type_spells["event_observed"] = (~tech_type_spells["still_used_2023"]).astype(
        int
    )


def plot_kaplan_meier_survival_curves_by_techtype_fo() -> None:
    plt.figure(figsize=(12, 7))
    kmf = KaplanMeierFitter()
    top_techtypes = tech_type_spells["techtype"].value_counts().head(8).index
    for techtype in top_techtypes:
        subset = tech_type_spells[tech_type_spells["techtype"] == techtype]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=techtype,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title("Technology Survival by Type (use == 1 only)")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Technology Type")
    plt.tight_layout()
    plt.show()


def custom_kaplan_meier_plot_with_formatting_improve() -> None:
    plt.figure(figsize=(12, 7))
    kmf = KaplanMeierFitter()
    top_techtypes = tech_type_spells["techtype"].value_counts().head(8).index
    for techtype in top_techtypes:
        subset = tech_type_spells[tech_type_spells["techtype"] == techtype]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=techtype,
        )
        kmf.plot_survival_function(ci_show=False)

    plt.title("Technology Survival by Type (use == 1 only)")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.xlim(0, 50)
    plt.ylim(0.2, 1.0)
    plt.yticks([0.25, 0.5, 0.75])
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.legend(title="Technology Type")
    plt.tight_layout()
    plt.show()


def compare_top_5_most_common_tech_types_pairwise() -> None:
    top_types = tech_type_spells["techtype"].value_counts().head(5).index
    tech_groups = {
        t: df
        for t, df in tech_type_spells[
            tech_type_spells["techtype"].isin(top_types)
        ].groupby("techtype")
    }
    results = []
    for a, b in itertools.combinations(top_types, 2):
        df1, df2 = (tech_groups[a], tech_groups[b])
        result = logrank_test(
            df1["duration_years"],
            df2["duration_years"],
            event_observed_A=df1["event_observed"],
            event_observed_B=df2["event_observed"],
        )
        results.append(
            (a, b, result.test_statistic, result.p_value, -np.log2(result.p_value))
        )

    pd.DataFrame(
        results, columns=["TechType 1", "TechType 2", "Chi2", "p-value", "-log2(p)"]
    )


def notebook_step_035() -> None:
    logrank_df


def label_rows_as_usa_or_other() -> None:
    tech_type_spells["group"] = tech_type_spells["ccode"].apply(
        lambda x: "USA" if x == 2 else "Other"
    )
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for group in ["USA", "Other"]:
        subset = tech_type_spells[tech_type_spells["group"] == group]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=group,
        )
        kmf.plot_survival_function(ci_show=True)

    plt.title("Technology Survival: USA vs. All Other Countries (use == 1 only)")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Group")
    plt.tight_layout()
    plt.show()


def notebook_step_037() -> None:
    top_types = df["techtype"].value_counts().head(5).index
    plt.figure(figsize=(12, 7))
    for techtype in top_types:
        subset = df[df["techtype"] == techtype]
        kmf.fit(subset["duration_years"], subset["event_observed"], label=techtype)
        kmf.plot_survival_function(ci_show=False)

    plt.title("Survival by Technology Type")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


def load_long_format_data() -> None:
    df = pd.read_csv("cow_arms_tech_long.csv")
    df = df[df["use"] == 1].copy()
    df = df.sort_values(["ccode", "techname", "year"])
    df["used_shift"] = df.groupby(["ccode", "techname"])["use"].shift(1, fill_value=0)
    df["new_spell"] = (df["use"] == 1) & (df["used_shift"] != 1)
    df["spell_id"] = df.groupby(["ccode", "techname"])["new_spell"].cumsum()
    spells = (
        df.groupby(["ccode", "techname", "techtype", "spell_id"])
        .agg(start_year=("year", "min"), end_year=("year", "max"))
        .reset_index()
    )
    spells["duration_years"] = spells["end_year"] - spells["start_year"] + 1
    spells["event_observed"] = (spells["end_year"] < 2023).astype(int)
    top_types = spells["techtype"].value_counts().head(5).index
    plt.figure(figsize=(12, 7))
    kmf = KaplanMeierFitter()
    for tech in top_types:
        subset = spells[spells["techtype"] == tech]
        kmf.fit(subset["duration_years"], subset["event_observed"], label=tech)
        kmf.plot_survival_function(ci_show=True)

    plt.title("Technology Survival by Type (use == 1 only)")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("km_techtype_top5.png")
    plt.close()
    results = []
    tech_groups = {
        t: df
        for t, df in spells[spells["techtype"].isin(top_types)].groupby("techtype")
    }
    for a, b in itertools.combinations(top_types, 2):
        df1, df2 = (tech_groups[a], tech_groups[b])
        result = logrank_test(
            df1["duration_years"],
            df2["duration_years"],
            event_observed_A=df1["event_observed"],
            event_observed_B=df2["event_observed"],
        )
        results.append(
            (a, b, result.test_statistic, result.p_value, -np.log2(result.p_value))
        )

    logrank_df = pd.DataFrame(
        results, columns=["TechType 1", "TechType 2", "Chi²", "p-value", "-log2(p)"]
    )
    logrank_df.to_csv("logrank_results_techtypes.csv", index=False)
    print("Log-rank test results saved to logrank_results_techtypes.csv")


def load_long_format_data_2() -> None:
    df = pd.read_csv("cow_arms_tech_long.csv")
    df = df[df["use"] == 1].copy()
    df = df.sort_values(["ccode", "techname", "year"])
    df["used_shift"] = df.groupby(["ccode", "techname"])["use"].shift(1, fill_value=0)
    df["new_spell"] = (df["use"] == 1) & (df["used_shift"] != 1)
    df["spell_id"] = df.groupby(["ccode", "techname"])["new_spell"].cumsum()
    spells = (
        df.groupby(["ccode", "techname", "techtype", "spell_id"])
        .agg(start_year=("year", "min"), end_year=("year", "max"))
        .reset_index()
    )
    spells["duration_years"] = spells["end_year"] - spells["start_year"] + 1
    spells["event_observed"] = (spells["end_year"] < 2023).astype(int)
    top_types = spells["techtype"].value_counts().head(5).index
    plt.figure(figsize=(12, 7))
    kmf = KaplanMeierFitter()
    for tech in top_types:
        subset = spells[spells["techtype"] == tech]
        kmf.fit(subset["duration_years"], subset["event_observed"], label=tech)
        kmf.plot_survival_function(ci_show=True)

    plt.title("Technology Survival by Type (use == 1 only)")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.xlim(0, 50)
    plt.ylim(0.2, 1.0)
    plt.yticks([0.25, 0.5, 0.75])
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig("km_techtype_top5_trimmed.png")
    plt.close()
    results = []
    tech_groups = {
        t: df
        for t, df in spells[spells["techtype"].isin(top_types)].groupby("techtype")
    }
    for a, b in itertools.combinations(top_types, 2):
        df1, df2 = (tech_groups[a], tech_groups[b])
        result = logrank_test(
            df1["duration_years"],
            df2["duration_years"],
            event_observed_A=df1["event_observed"],
            event_observed_B=df2["event_observed"],
        )
        results.append(
            (a, b, result.test_statistic, result.p_value, -np.log2(result.p_value))
        )

    logrank_df = pd.DataFrame(
        results, columns=["TechType 1", "TechType 2", "Chi²", "p-value", "-log2(p)"]
    )
    logrank_df.to_csv("logrank_results_techtypes.csv", index=False)
    print("Log-rank test results saved to logrank_results_techtypes.csv")


def filter_to_only_small_arms_usage_spells() -> None:
    small_arms_spells = spells[spells["techtype"] == "Small arms"].copy()
    small_arms_spells["continent"] = small_arms_spells["ccode"].apply(get_continent)
    valid_continents = ["Africa", "Asia", "Europe", "Americas", "Middle East"]
    small_arms_spells = small_arms_spells[
        small_arms_spells["continent"].isin(valid_continents)
    ]
    plt.figure(figsize=(12, 7))
    kmf = KaplanMeierFitter()
    for continent in sorted(small_arms_spells["continent"].unique()):
        subset = small_arms_spells[small_arms_spells["continent"] == continent]
        kmf.fit(
            subset["duration_years"],
            event_observed=subset["event_observed"],
            label=continent,
        )
        kmf.plot_survival_function(ci_show=True)

    plt.title("Small Arms Survival by Continent (use == 1 only)")
    plt.xlabel("Years of Use")
    plt.ylabel("Survival Probability")
    plt.xlim(0, 50)
    plt.ylim(0.2, 1.0)
    plt.yticks([0.25, 0.5, 0.75])
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(title="Continent")
    plt.tight_layout()
    plt.show()


def prepare_the_data_for_cox_proportional_hazards_mo() -> None:
    cox_df = spells[["duration_years", "event_observed", "techname"]].dropna().copy()
    cox_df = pd.get_dummies(cox_df, columns=["techname"], drop_first=True)
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration_years", event_col="event_observed")
    cph.print_summary()


def notebook_step_042() -> None:
    cox_df = spells[["duration_years", "event_observed", "techtype"]].dropna().copy()
    cox_df = pd.get_dummies(cox_df, columns=["techtype"], drop_first=True)
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration_years", event_col="event_observed")
    cph.print_summary()


def filter_the_spells_to_only_small_arms_and_fighter() -> None:
    filtered_spells = spells[
        spells["techtype"].isin(["Small arms", "Fighter aircraft"])
    ].copy()
    filtered_spells["is_fighter"] = (
        filtered_spells["techtype"] == "Fighter aircraft"
    ).astype(int)
    cox_df = filtered_spells[["duration_years", "event_observed", "is_fighter"]]
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration_years", event_col="event_observed")
    cph.print_summary()


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


if __name__ == "__main__":
    main()
