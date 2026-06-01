"""Notebook steps (auto-split)."""


def label_us_period(row):
    if row["ccode"] != 2:
        return None
    if row["start_year"] < 1916:
        return "1816–1915"
    elif row["start_year"] < 1941:
        return "1916–1940"
    elif row["start_year"] <= 1991:
        return "1945–1991"
    return "1992+"
