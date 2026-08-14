"""oed.entry.lemma computation — is a headword already its own base form?"""

from __future__ import annotations

import pytest

from concordance.oed.lemma import _normalize_pos, is_lemma
from concordance.tokenize import load_nlp


def test_normalize_pos_single_tokens():
    assert _normalize_pos("sb") == ({"NOUN"}, False)
    assert _normalize_pos("v") == ({"VERB"}, False)
    assert _normalize_pos("a") == ({"ADJ"}, False)
    assert _normalize_pos("adv") == ({"ADV"}, False)


def test_normalize_pos_case_and_ocr_noise():
    # OED OCR yields case variants (A/N/V) and stray-period typos (ppL a).
    assert _normalize_pos("A") == ({"ADJ"}, False)
    assert _normalize_pos("N") == ({"NOUN"}, False)
    assert _normalize_pos("V") == ({"VERB"}, False)


def test_normalize_pos_historical_verb_form_bigrams():
    assert _normalize_pos("ppl. a") == (set(), True)
    assert _normalize_pos("pa. pple") == (set(), True)
    assert _normalize_pos("pa. t") == (set(), True)
    # a historical tag alongside another category still forces the
    # historical (VERB) reading — the whole point of these tags.
    cats, historical = _normalize_pos("v Pa. t pa. pple")
    assert historical is True


def test_normalize_pos_multi_token():
    assert _normalize_pos("a sb") == ({"ADJ", "NOUN"}, False)


def test_normalize_pos_empty_or_none():
    assert _normalize_pos(None) == (set(), False)
    assert _normalize_pos("") == (set(), False)
    assert _normalize_pos("TBD") == (set(), False)


@pytest.fixture(scope="module")
def nlp():
    return load_nlp()


def test_participial_adjective_collapses_to_verb_root(nlp):
    # OED gives these their own headword entry, but for lemma purposes they
    # are inflected forms of the base verb, not standalone vocabulary.
    assert is_lemma(nlp, "abandoned", "ppl. a") is False
    assert is_lemma(nlp, "alcoholized", "ppl. a") is False
    assert is_lemma(nlp, "accommodating", "ppl. a") is False


def test_base_verb_is_its_own_lemma(nlp):
    assert is_lemma(nlp, "abandon", "v") is True
    assert is_lemma(nlp, "diagnose", "v") is True


def test_plural_noun_is_not_a_lemma(nlp):
    assert is_lemma(nlp, "cats", "sb") is False
    assert is_lemma(nlp, "cat", "sb") is True


def test_comparative_adjective_is_not_a_lemma(nlp):
    assert is_lemma(nlp, "faster", "a") is False


def test_multi_pos_true_only_if_every_reading_is_unchanged(nlp):
    # "abstract" is citation-form as both adjective and noun.
    assert is_lemma(nlp, "abstract", "a sb") is True
    # "greater" is a comparative under the adjective reading, so the
    # multi-tag entry as a whole is not a lemma.
    assert is_lemma(nlp, "greater", "a sb") is False


def test_null_pos_falls_back_to_context_free_spacy_guess(nlp):
    # No OED tag at all — spaCy's own guess still reduces a clear -ing form.
    assert is_lemma(nlp, "running", None) is False
    # An unfamiliar/archaic headword spaCy has no reason to alter.
    assert is_lemma(nlp, "cheirokinaesthesia", None) is True
