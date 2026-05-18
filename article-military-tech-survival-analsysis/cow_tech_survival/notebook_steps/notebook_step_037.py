"""Notebook steps (auto-split)."""

import matplotlib.pyplot as plt


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
