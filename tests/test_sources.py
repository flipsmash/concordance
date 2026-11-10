"""Readable definition sources (concordance/sources.py)."""

import pytest

from concordance.sources import describe_definition_source as describe


@pytest.mark.parametrize("raw, expected", [
    ("Local Wiktionary (DB)", ("Wiktionary", None)),
    ("OED", ("0 Dict", None)),
    ("oed", ("0 Dict", None)),
    ("mw", ("Merriam-Webster", None)),
    ("merriam_webster", ("Merriam-Webster", None)),
    ("dm", ("Datamuse", None)),
    ("Wordnik (century)", ("Century Dictionary", None)),
    ("Web (LLM-extracted)", ("Web search", None)),
    ("Local Wiktionary (DB) (synonym of 'odoriferous')", ("Wiktionary", "the definition of its synonym “odoriferous”")),
    ("derived (stringless + -ly)", ("Derived", "built from stringless + -ly")),
    ("derived (inflection of foreknow)", ("Derived", "a form of “foreknow”")),
    ("Web (manual): variant of cleidomancy", ("Web (added by hand)", "variant of cleidomancy")),
    ("Some New Source", ("Some New Source", None)),        # unknown: shown as written
    ("", (None, None)), (None, (None, None)), ("corpus", (None, None)),
])
def test_describe(raw, expected):
    assert describe(raw) == expected
