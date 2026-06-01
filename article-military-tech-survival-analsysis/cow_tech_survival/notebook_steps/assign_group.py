"""Notebook steps (auto-split)."""


def assign_group(ccode):
    if ccode in china_ccode:
        return "China"
    elif ccode in russia_ccodes:
        return "Russia/USSR"
    elif ccode in nato_ccodes:
        return "NATO"
    return None
