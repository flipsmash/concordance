"""Lemma resolution — guarding against spaCy stripping real words to non-words.

Uses a duck-typed token so no spaCy model load is needed; _attested is real
(wordfreq + the NLTK wordlist)."""

from __future__ import annotations

from concordance.tokenize import _resolve_lemma


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
