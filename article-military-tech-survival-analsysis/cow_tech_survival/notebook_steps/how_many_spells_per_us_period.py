"""Notebook steps (auto-split)."""


def how_many_spells_per_us_period() -> None:
    # How many spells per US period?
    us_period_counts = spell_df.query("ccode == 2")["start_year"].value_counts()
    print(us_period_counts)
