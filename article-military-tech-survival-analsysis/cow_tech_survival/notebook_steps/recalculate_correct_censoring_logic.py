"""Notebook steps (auto-split)."""

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


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
