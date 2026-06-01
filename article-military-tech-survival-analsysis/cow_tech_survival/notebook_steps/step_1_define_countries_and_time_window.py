"""Notebook steps (auto-split)."""

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


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
