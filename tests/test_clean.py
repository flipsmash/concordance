"""clean.py's punctuation normalization, especially the possessive-apostrophe
fix: U+02BC (MODIFIER LETTER APOSTROPHE) must become a straight apostrophe
before tokenization, or spaCy treats "publisherʼs" as one alphabetic token
instead of splitting off the possessive "'s" (see clean.py's _PUNCT comment)."""

from __future__ import annotations

import pytest

from concordance.clean import clean
from concordance.extract import Chapter
from concordance.tokenize import load_nlp, tokenize


def test_modifier_letter_apostrophe_normalized_to_straight_quote():
    assert clean("the publisherʼs office") == "the publisher's office"


def test_curly_and_straight_apostrophes_still_normalized():
    assert clean("the publisher’s office") == "the publisher's office"
    assert clean("the publisher's office") == "the publisher's office"


@pytest.fixture(scope="module")
def nlp():
    return load_nlp()


def test_possessive_with_modifier_letter_apostrophe_does_not_become_a_word(nlp):
    text = clean("The publisherʼs office was closed. The bankerʼs house stood near.")
    cands = tokenize([Chapter(title="1", text=text)], nlp=nlp)
    assert "publisherʼs" not in cands
    assert "bankerʼs" not in cands
    assert "publisher" in cands
    assert "banker" in cands
