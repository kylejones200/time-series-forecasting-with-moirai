"""Notebook steps (auto-split)."""


def label_us_time_bin(row):
    if row["ccode"] != 2:
        return None
    elif 1923 <= row["start_year"] < 1973:
        return "1923–1973"
    elif 1973 <= row["start_year"] <= 2023:
        return "1973–2023"
    return None
