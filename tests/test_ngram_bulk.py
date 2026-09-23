"""Bulk Google Books Ngram parsing (pure; tiny synthetic shard)."""
from __future__ import annotations

import gzip

from concordance import ngram_bulk as nb


def _totals():
    return {y: 1_000_000 for y in range(nb.FIRST_YEAR, nb.LAST_YEAR + 1)}


def test_read_totals(tmp_path):
    p = tmp_path / "totalcounts-1"
    p.write_text(" \t1800,500,10,1\t2019,900,20,2\n")
    assert nb.read_totals(p) == {1800: 500, 2019: 900}


def test_iter_shard_keeps_only_untagged_lowercase_words(tmp_path):
    p = tmp_path / "1-00000-of-00024.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write("copperware\t1885,10,2\t2010,3,1\n")
        f.write("copperware_NOUN\t1885,10,2\n")          # POS-tagged duplicate
        f.write("Copperware\t1900,5,1\n")                # capitalized: API queried lowercase
        f.write("1,387,805_NUM\t1907,1,1\n")
        f.write("cortège\t1900,4,2\n")                   # UTF-8 accented word kept
        f.write("well-known\t1950,7,3\n")
    got = dict(nb.iter_shard(p))
    assert got == {"copperware": {1885: 10, 2010: 3}, "cortège": {1900: 4}, "well-known": {1950: 7}}


def test_features_match_viewer_smoothing():
    counts = {2019: 70}                                  # a single spike at the last year
    f = nb.features(counts, _totals())
    # smoothing=3 window at the range end is 2016..2019 (4 years): peak there = 70/1e6/4
    assert abs(f["peak"] - 70 / 1e6 / 4) < 1e-15 and f["peak_year"] in (2016, 2017, 2018, 2019)
    assert f["recent"] > 0 and 0 < f["recency_ratio"] <= 1


def test_features_absent_word_is_zero_not_missing():
    f = nb.features({}, _totals())
    assert f == {"peak": 0.0, "recent": 0.0, "recency_ratio": None, "peak_year": None}


def test_decade_counts_skip_pre_1800():
    bins = nb.decade_counts({1650: 99, 1805: 2, 1809: 3, 2015: 7})
    assert len(bins) == len(nb.DECADES) and bins[0] == 5 and bins[-1] == 7 and sum(bins) == 12
