"""Notebook steps (auto-split)."""


def notebook_step_017() -> None:
    us_subset = arms_df[arms_df["ccode"] == 2]
    us_subset.groupby("techname")["total_use"].nunique().sort_values()
