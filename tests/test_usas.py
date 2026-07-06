"""USAS tagset integrity — the category backbone must parse into a sound tree."""

from __future__ import annotations

from concordance import usas


def test_counts_match_published_tagset():
    cats = usas.categories()
    assert len(cats) == 253
    assert sum(1 for c in cats if c["parent_code"] is None) == 21   # top-level fields


def test_every_parent_link_resolves():
    cats = usas.categories()
    codes = {c["code"] for c in cats}
    for c in cats:
        if c["parent_code"] is not None:
            assert c["parent_code"] in codes, c


def test_parent_derivation_skips_missing_tiers():
    by = {c["code"]: c for c in usas.categories()}
    assert by["A1.5.2"]["parent_code"] == "A1.5"
    assert by["A5.1"]["parent_code"] == "A5"
    assert by["A1"]["parent_code"] == "A"
    assert by["A"]["parent_code"] is None
    assert by["S1.1.1"]["parent_code"] == "S1.1"
    # G3's only explicit ancestor is the top field G
    assert by["G3"]["parent_code"] == "G"


def test_levels_are_consistent():
    for c in usas.categories():
        if c["parent_code"] is None:
            assert c["level"] == 0
        else:
            assert c["level"] == c["code"].count(".") + 1


def test_operational_z_bins_not_assignable():
    by = {c["code"]: c for c in usas.categories()}
    for junk in ("Z4", "Z5", "Z8", "Z9", "Z99"):
        assert by[junk]["assignable"] is False
    assert by["Z1"]["assignable"] is True         # personal names is a real category
    assert by["X3.2"]["assignable"] is True


def test_codes_unique():
    codes = [c["code"] for c in usas.categories()]
    assert len(codes) == len(set(codes))
