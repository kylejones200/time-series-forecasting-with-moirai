"""Notebook steps (auto-split)."""


def notebook_step_016() -> None:
    arms_df["used"] = arms_df["total_use"].notna() & (arms_df["total_use"] > 0)
