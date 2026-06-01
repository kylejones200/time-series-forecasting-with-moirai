"""Notebook steps (auto-split)."""

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


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
