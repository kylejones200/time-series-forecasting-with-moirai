"""Notebook steps (auto-split)."""


def label_nato_group(ccode):
    if ccode == 2:
        return "USA"
    elif ccode == 20:
        return "Canada"
    elif ccode == 220:
        return "France"
    elif ccode == 200:
        return "UK"
    elif ccode in nato_ccodes:
        return "Other NATO"
    return None
