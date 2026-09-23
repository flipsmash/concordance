"""Ex-ante difficulty scalar (pure)."""
from __future__ import annotations
from concordance import difficulty as d


def test_rarer_is_harder():
    common = d.score(3.4, 1e-6)[0]
    rare = d.score(1.0, 1e-8)[0]
    assert rare > common


def test_ngram_breaks_ties_among_zipf_zero_words():
    # both unlisted by wordfreq (zipf 0) but different recent print frequency
    less = d.score(0.0, 1e-8)[0]      # still shows up in print
    more = d.score(0.0, 1e-10)[0]     # barely in print
    assert more > less


def test_recent_print_not_peak_drives_rarity():
    # the live bug: cognoscibility's 1674 spike (peak) outranked copperware,
    # which is ~8x more common in 2000-2019 print. Peak must not matter.
    copperware = d.score(0.0, 9.5e-9, ngram_peak=4.9e-8)[0]
    cognosc = d.score(0.0, 1.2e-9, ngram_peak=3.3e-8)[0]
    assert cognosc > copperware
    assert d.score(0.0, 1e-9, ngram_peak=1e-5)[0] == d.score(0.0, 1e-9, ngram_peak=1e-9)[0]


def test_zipf_and_rarity_move_together():
    # zipf in the factors is the unified value rarity comes from -- a zipf-0
    # word reports its print-derived zipf, not wordfreq's "not listed" 0.
    _, f = d.score(0.0, 1e-9)
    assert f["zipf_source"] == "ngram" and f["zipf_web"] == 0.0
    assert abs(f["zipf"] - 0.0) < 1e-9                    # 1e-9 * 1e9 = 1 -> log10 = 0
    lo = d.score(0.0, 1e-8)[1]
    assert lo["zipf"] > f["zipf"] and lo["rarity"] < f["rarity"]


def test_wordfreq_wins_when_listed_and_root_when_higher():
    assert d.unified_zipf(1.5, 1e-7) == (1.5, "wordfreq")
    assert d.unified_zipf(0.0, 1e-7)[1] == "ngram"
    assert d.unified_zipf(0.0, None) == (d._ZIPF_FLOOR, "unseen")
    assert d.unified_zipf(0.0, 0.0) == (d._ZIPF_FLOOR, "ngram")
    assert d.unified_zipf(0.0, 1e-9, root_zipf=4.2) == (4.2, "root")


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
    sc, f = d.score(0.0, 0.0, ngram_peak=0.0, archaic="obsolete", archaic_conf=1.0, has_domain=True)
    assert 0 <= sc <= 100
    assert "why" in f and "rarity" in f
    assert "absent from print" in f["why"]
