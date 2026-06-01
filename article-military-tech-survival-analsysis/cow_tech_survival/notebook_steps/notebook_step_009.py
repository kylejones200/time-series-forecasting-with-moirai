"""Notebook steps (auto-split)."""


def notebook_step_009() -> None:
    spell_df["us_period"] = spell_df.apply(label_us_period, axis=1)
