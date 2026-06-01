"""Notebook steps (auto-split)."""


def get_continent(ccode):
    if ccode in africa_codes:
        return "Africa"
    elif ccode in asia_codes:
        return "Asia"
    elif ccode in europe_codes:
        return "Europe"
    elif ccode in americas_codes:
        return "Americas"
    elif ccode in middle_east_codes:
        return "Middle East"
    return "Other"
