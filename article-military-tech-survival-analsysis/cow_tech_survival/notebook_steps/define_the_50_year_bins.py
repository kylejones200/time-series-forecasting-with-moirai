"""Notebook steps (auto-split)."""

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


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
