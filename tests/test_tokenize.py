"""Lemma resolution — guarding against spaCy stripping real words to non-words.

Uses a duck-typed token so no spaCy model load is needed; _attested is real
(wordfreq + the NLTK wordlist)."""

from __future__ import annotations

import pytest

from concordance.extract import Chapter
from concordance.tokenize import _resolve_lemma, load_nlp, tokenize


class _Tok:
    def __init__(self, text, lemma):
        self.text = text
        self.lemma_ = lemma


def test_reverts_lemma_that_is_a_nonword():
    # spaCy mangles these; the surface form is the real (archaic) word.
    assert _resolve_lemma(_Tok("afeared", "afeare")) == "afeared"
    assert _resolve_lemma(_Tok("overscutched", "overscutche")) == "overscutched"
    assert _resolve_lemma(_Tok("windring", "windre")) == "windring"


def test_keeps_a_valid_lemma():
    assert _resolve_lemma(_Tok("besmirching", "besmirch")) == "besmirch"
    assert _resolve_lemma(_Tok("running", "run")) == "run"
    assert _resolve_lemma(_Tok("daisies", "daisy")) == "daisy"


def test_unchanged_token_passes_through():
    assert _resolve_lemma(_Tok("cangue", "cangue")) == "cangue"


def test_both_unattested_keeps_spacy_lemma():
    # a genuinely coined pair — nothing better to fall back to, keep spaCy's guess
    assert _resolve_lemma(_Tok("zzblorping", "zzblorp")) == "zzblorp"


def test_empty_lemma_falls_back_to_surface():
    assert _resolve_lemma(_Tok("Foo", "")) == "foo"


# --- cap_ratio: ALL-CAPS boilerplate must not count as name-capitalization ---
# Regression: "MERCHANTABILITY" (Gutenberg's standard license boilerplate,
# present in virtually every book in this corpus) and similar ALL-CAPS text
# were counted the same as Title-Case ("Bloom") capitalization, driving
# cap_ratio high enough to get ordinary dictionary words rejected as proper
# nouns. Fixed in tokenize.py's mid-sentence counting, not propernouns.py's
# threshold -- the cap_ratio signal itself must stay untouched for the
# Bloom/Baker case (a common word consistently Title-Case-capitalized
# throughout one book really is being used as a name there).

@pytest.fixture(scope="module")
def nlp():
    return load_nlp()


def _cap_ratio(nlp, text, lemma):
    cands = tokenize([Chapter(title="1", text=text)], nlp=nlp)
    return cands[lemma].cap_ratio


def test_all_caps_occurrence_does_not_count_as_capitalized(nlp):
    # Every mid-sentence occurrence of "warranty" here is ALL-CAPS -- none of
    # it should count as name-style capitalization evidence.
    text = "The buyer saw WARRANTY posted on the wall. She read WARRANTY twice more."
    assert _cap_ratio(nlp, text, "warranty") == 0.0


def test_all_caps_occurrence_is_excluded_not_counted_against():
    nlp = load_nlp()
    # One ALL-CAPS occurrence (excluded entirely) plus one genuine mid-sentence
    # Title-Case occurrence -- the ratio must reflect only the real evidence
    # (1/1 = 1.0), not be diluted by the ALL-CAPS occurrence counting as a
    # non-capitalized data point (which would wrongly give 1/2 = 0.5).
    text = "The clerk saw WARRANTY on the wall. Later she found a Warranty in the drawer."
    assert _cap_ratio(nlp, text, "warranty") == 1.0


def test_title_case_capitalization_still_counts(nlp):
    # Regression guard: the ALL-CAPS exclusion must not touch ordinary
    # Title-Case name detection (the Bloom/Baker collision case).
    text = "The sailor met Bloom at the docks. Later Bloom returned alone."
    assert _cap_ratio(nlp, text, "bloom") == 1.0
