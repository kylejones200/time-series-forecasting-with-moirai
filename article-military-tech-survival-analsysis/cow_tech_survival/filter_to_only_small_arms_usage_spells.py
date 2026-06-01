"""Auto-split from legacy monolithic script."""

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


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
