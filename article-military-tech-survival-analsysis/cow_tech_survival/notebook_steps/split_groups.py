"""Notebook steps (auto-split)."""

from lifelines.statistics import logrank_test


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
