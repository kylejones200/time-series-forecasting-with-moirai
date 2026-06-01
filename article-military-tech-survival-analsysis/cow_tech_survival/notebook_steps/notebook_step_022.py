"""Notebook steps (auto-split)."""


def notebook_step_022() -> None:
    us_spells.groupby("us_period")["event_observed"].value_counts()
