"""Notebook steps (auto-split)."""

from lifelines import CoxPHFitter


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
