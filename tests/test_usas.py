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


# --- subtree_match / subtree_sql / children_of / by_code ------------------
# (the 4-level Categories drilldown's boundary rule and query helpers)

# All 16 real sibling-collision pairs in the tagset: a naive `other.startswith(code)`
# prefix check gets every one of these wrong (A1/A10..A15 are siblings under "A",
# not parent/child; same for Z9/Z99), found by an exhaustive check during design.
_SIBLING_COLLISIONS = [
    ("A1", "A10"), ("A1", "A11"), ("A1", "A11.1"), ("A1", "A11.2"),
    ("A1", "A12"), ("A1", "A13"), ("A1", "A13.1"), ("A1", "A13.2"),
    ("A1", "A13.3"), ("A1", "A13.4"), ("A1", "A13.5"), ("A1", "A13.6"),
    ("A1", "A13.7"), ("A1", "A14"), ("A1", "A15"), ("Z9", "Z99"),
]


def test_subtree_match_rejects_sibling_collisions():
    for code, other in _SIBLING_COLLISIONS:
        assert other.startswith(code)  # the naive check a real bug would use
        assert not usas.subtree_match(code, other), (code, other)


def test_subtree_match_exhaustive_against_real_tagset():
    codes = [c["code"] for c in usas.categories()]
    known_collisions = set(_SIBLING_COLLISIONS)
    for code in codes:
        for other in codes:
            naive = other.startswith(code)
            real = usas.subtree_match(code, other)
            if code == other:
                assert real
            elif naive and (code, other) in known_collisions:
                assert not real
            else:
                assert real == naive, (code, other)


def test_subtree_sql_agrees_with_subtree_match():
    import re

    codes = [c["code"] for c in usas.categories()]
    for code in codes:
        exact, like = usas.subtree_sql(code)
        assert exact == code
        pattern = "^" + re.escape(like).replace("%", ".*") + "$"
        for other in codes:
            like_match = bool(re.match(pattern, other)) or other == exact
            assert like_match == usas.subtree_match(code, other), (code, other)


def test_children_of():
    assert {c["code"] for c in usas.children_of("I2")} == {"I2.1", "I2.2"}
    assert usas.children_of("I2.2") == []  # a real leaf


def test_children_of_level3():
    # Only 5 of 100 level-2 codes have any level-3 children at all.
    assert {c["code"] for c in usas.children_of("S1.1")} == {"S1.1.1", "S1.1.2", "S1.1.3", "S1.1.4"}
    assert usas.children_of("I2.2") == []


def test_children_of_a1_mixes_absolute_depths():
    # A1's numbering implies an unnamed "A1.1" tier that was never given its
    # own row, so A1's direct children legitimately mix level-2 codes (A1.2,
    # A1.5, ...) with two direct level-3 codes (A1.1.1, A1.1.2).
    children = {c["code"]: c["level"] for c in usas.children_of("A1")}
    assert children["A1.1.1"] == 3
    assert children["A1.1.2"] == 3
    assert children["A1.5"] == 2


def test_every_level0_code_has_a_child():
    for c in usas.categories():
        if c["level"] == 0:
            assert usas.children_of(c["code"]), c["code"]


def test_by_code():
    assert usas.by_code("A1.5.2") == {
        "code": "A1.5.2", "name": "Usefulness", "parent_code": "A1.5",
        "level": 3, "assignable": True,
    }
    assert usas.by_code("not-a-real-code") is None
