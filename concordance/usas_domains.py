"""Macro-domain color buckets for the word-similarity graph (§ visualization).

A presentation-layer grouping *on top of* usas.py's real 21-field taxonomy, not
a replacement for it — kept in a separate module so it's never mistaken for
taxonomically authoritative. A validated CVD-safe categorical palette only
holds 6 distinct hues for a scatter-like layout (any two nodes can end up
adjacent, so the all-pairs colorblindness check applies, not just adjacent-
pairs), so the 21 top-level USAS fields compress into 6 buckets here for node
color only. The specific USAS code/name is still surfaced separately (see the
graph endpoint's GraphNode.usas_code/usas_name) — compressing to 6 buckets
only affects color, never the information shown in a tooltip.

This grouping is a starting default, not a linguistic claim — it's fine to
rebalance later; adjusting DOMAIN_BUCKETS is the only change needed.
"""

from __future__ import annotations

DOMAIN_BUCKETS = {
    "mind_language":       {"name": "Mind, Language & Learning",     "codes": ["A", "P", "Q", "X"]},
    "people_society":      {"name": "People & Society",              "codes": ["B", "S", "G", "Z"]},
    "emotion_leisure":     {"name": "Emotion & Leisure",              "codes": ["E", "K"]},
    "nature_science":      {"name": "Nature & Science",               "codes": ["F", "L", "W", "Y"]},
    "making_materials":    {"name": "Making & Materials",             "codes": ["C", "H", "O"]},
    "time_space_commerce": {"name": "Time, Space, Number & Commerce", "codes": ["M", "N", "T", "I"]},
}

_CODE_TO_BUCKET = {code: key for key, v in DOMAIN_BUCKETS.items() for code in v["codes"]}


def bucket_for(usas_top_code: str | None) -> str | None:
    """The color bucket key for a top-level USAS code (e.g. "S" -> "people_society"),
    or None if the code is unrecognized/absent — callers render that as gray."""
    if not usas_top_code:
        return None
    return _CODE_TO_BUCKET.get(usas_top_code)


def legend_entries() -> list[dict]:
    """(bucket key, display name) for every bucket, in fixed display order —
    the "Uncategorized"/gray entry is a client-side-only concept, not included
    here (it's not a real USAS bucket)."""
    return [{"bucket": key, "name": v["name"]} for key, v in DOMAIN_BUCKETS.items()]
