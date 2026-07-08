"""Ex-ante difficulty scalar (pure)."""
from __future__ import annotations
from concordance import difficulty as d


def test_rarer_is_harder():
    common = d.score(3.4, 1e-6)[0]
    rare = d.score(1.0, 1e-8)[0]
    assert rare > common


def test_ngram_breaks_ties_among_zipf_zero_words():
    # both absent from web (zipf 0) but different print rarity -> different scores
    less = d.score(0.0, 1e-6)[0]      # still shows up in print
    more = d.score(0.0, 1e-10)[0]     # barely in print
    assert more > less


def test_obsolete_adds_difficulty_weighted_by_confidence():
    base = d.score(1.5, 1e-7)[0]
    hi = d.score(1.5, 1e-7, archaic="obsolete", archaic_conf=0.95)[0]
    lo = d.score(1.5, 1e-7, archaic="obsolete", archaic_conf=0.5)[0]
    assert hi > lo > base                     # confidence scales the contribution


def test_transparent_morphology_eases():
    opaque = d.score(1.5, 1e-7)[0]
    transparent = d.score(1.5, 1e-7, morph_transparent=True)[0]
    assert transparent < opaque


def test_score_bounded_and_reports_why():
    sc, f = d.score(0.0, 0.0, archaic="obsolete", archaic_conf=1.0, has_domain=True)
    assert 0 <= sc <= 100
    assert "why" in f and "rarity" in f
