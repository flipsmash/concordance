"""archive_metadata: Gutenberg-boilerplate stripping, word stats, and
publication year/era extraction — all pure logic, no DB/network needed
(fetch_publication_info's own network call is exercised live via the CLI,
not unit-tested here)."""

from concordance.archive_metadata import (
    _ERA_RE,
    _YEAR_RE,
    extract_gutenberg_id,
    strip_gutenberg_boilerplate,
    word_stats,
    year_to_era,
)

_HEADER = """The Project Gutenberg eBook of Some Book

Release date: September 17, 2004 [eBook #3190]

Language: English

*** START OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***
"""

_FOOTER = """
*** END OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***

More boilerplate about licenses and mirrors goes here.
"""


def test_strip_gutenberg_boilerplate_keeps_only_the_body():
    text = _HEADER + "The quick brown fox jumps over the lazy dog." + _FOOTER
    stripped = strip_gutenberg_boilerplate(text)
    assert stripped.strip() == "The quick brown fox jumps over the lazy dog."


def test_strip_gutenberg_boilerplate_falls_back_to_full_text_without_markers():
    text = "Just some plain text with no Gutenberg markers at all."
    assert strip_gutenberg_boilerplate(text) == text


def test_strip_gutenberg_boilerplate_falls_back_to_start_only_when_end_missing():
    text = _HEADER + "Body text here."
    stripped = strip_gutenberg_boilerplate(text)
    assert stripped.strip() == "Body text here."


def test_strip_gutenberg_boilerplate_handles_an_indented_end_marker():
    # Regression: confirmed live on ~54 real corpus files whose END marker
    # is indented ("            *** END OF ... ***") -- a bare ^\*\*\* anchor
    # never matched, silently leaving the whole license-footer text inside
    # word_stats for every one of those books.
    indented_footer = "\n            *** END OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***\n\nMore boilerplate.\n"
    text = _HEADER + "The quick brown fox jumps over the lazy dog." + indented_footer
    stripped = strip_gutenberg_boilerplate(text)
    assert stripped.strip() == "The quick brown fox jumps over the lazy dog."


def test_strip_gutenberg_boilerplate_uses_last_start_when_header_is_duplicated():
    # Regression: confirmed live on 2 real corpus files (a Gutenberg
    # re-release that repeats its whole Title:/Author:/Release date: block
    # -- each ending in its own START line) -- the FIRST start left the
    # second header+marker embedded in the "stripped" text.
    text = (
        _HEADER
        + _HEADER  # the whole header+START repeats verbatim before the real body
        + "The quick brown fox jumps over the lazy dog."
        + _FOOTER
    )
    stripped = strip_gutenberg_boilerplate(text)
    assert stripped.strip() == "The quick brown fox jumps over the lazy dog."


def test_strip_gutenberg_boilerplate_uses_last_end_to_avoid_truncating_a_compiled_work():
    # Regression: a book-merge "(Complete)" compile can carry an
    # intermediate part's own END marker mid-body (that part's marker went
    # unmatched at compile time, so split_gutenberg_parts folded its footer
    # into "body"). Using the FIRST end here silently discarded the entire
    # rest of the work -- confirmed live, lost all of a 2-part play's second
    # half. The last END must always be treated as the true final boundary.
    text = (
        _HEADER
        + "Part one text."
        + _FOOTER
        + "Part two text, which must survive."
        + _FOOTER
    )
    stripped = strip_gutenberg_boilerplate(text)
    assert "Part one text." in stripped
    assert "Part two text, which must survive." in stripped
    assert stripped.strip().endswith("Part two text, which must survive.")


def test_extract_gutenberg_id_finds_the_ebook_number():
    assert extract_gutenberg_id(_HEADER) == 3190


def test_extract_gutenberg_id_returns_none_when_absent():
    assert extract_gutenberg_id("No id here.") is None


def test_word_stats_counts_total_and_distinct_nonstop():
    # "the"/"over" are stopwords; "quick"/"brown"/"fox"/"jumps"/"lazy"/"dog"
    # aren't. "the" appears twice, contributing 2 to the total but only
    # ever counted once (or not at all) toward the distinct-nonstop set.
    text = "The quick brown fox jumps over the lazy dog."
    total, distinct_nonstop = word_stats(text)
    assert total == 9  # The, quick, brown, fox, jumps, over, the, lazy, dog
    assert distinct_nonstop == 6  # quick, brown, fox, jumps, lazy, dog


def test_word_stats_is_case_insensitive_for_distinct_count():
    text = "Fox fox FOX"
    total, distinct_nonstop = word_stats(text)
    assert total == 3
    assert distinct_nonstop == 1


def test_year_regex_extracts_exact_year_near_a_reporting_verb():
    summary = '"Moby Dick" by Herman Melville is an epic novel published in 1851. Sailor Ishmael narrates.'
    m = _YEAR_RE.search(summary)
    assert m.group(1) == "1851"


def test_year_regex_does_not_match_an_unrelated_number():
    summary = "This book was digitized by 12 volunteers over several months in a library."
    assert _YEAR_RE.search(summary) is None


def test_era_regex_extracts_century_hedge():
    summary = "This children's book is written in the early 20th century, exploring rural life."
    m = _ERA_RE.search(summary)
    assert m.group(1) == "early 20th century"


def test_era_regex_extracts_decade_hedge():
    summary = "This science fiction novel was written in the early 1950s, exploring space travel."
    m = _ERA_RE.search(summary)
    assert m.group(1) == "early 1950s"


def test_era_regex_does_not_fire_when_year_regex_already_matched():
    # Both patterns CAN independently match the same summary; the caller
    # (fetch_publication_info) is what decides year wins over era, not the
    # regexes themselves -- this just confirms era still parses correctly
    # even when an exact year is also present, so that ordering logic has
    # real values to choose between.
    summary = '"Middlemarch" by George Eliot is a novel published in 1871-1872.'
    assert _YEAR_RE.search(summary).group(1) == "1871"


def test_year_to_era_early_mid_late_thirds():
    assert year_to_era(1905) == "early 20th century"   # the user's own example
    assert year_to_era(1901) == "early 20th century"
    assert year_to_era(1933) == "early 20th century"
    assert year_to_era(1934) == "mid 20th century"
    assert year_to_era(1966) == "mid 20th century"
    assert year_to_era(1967) == "late 20th century"
    assert year_to_era(2000) == "late 20th century"    # last year of the 20th century


def test_year_to_era_century_boundary_and_ordinal_suffix():
    assert year_to_era(2001) == "early 21st century"   # first year of the 21st
    assert year_to_era(1801) == "early 19th century"
    assert year_to_era(1650) == "mid 17th century"
    assert year_to_era(1667) == "late 17th century"


def test_year_to_era_avoids_11th_12th_13th_ordinal_mistakes():
    assert year_to_era(1005) == "early 11th century"
    assert year_to_era(1105) == "early 12th century"
    assert year_to_era(1205) == "early 13th century"
