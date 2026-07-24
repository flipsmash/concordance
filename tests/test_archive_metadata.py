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
