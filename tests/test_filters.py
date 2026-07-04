"""Regression tests for the deterministic filters — the parts most likely to
silently over- or under-drop. No spaCy or model needed."""

from wordfreq import zipf_frequency

from concordance.config import Config
from concordance.floor import apply_floor
from concordance.model import Candidate, Occurrence, RejectReason, Verdict
from concordance.propernouns import strip_proper_nouns
from concordance.validity import ValidityGate


def _cand(lemma, *, zipf=None, count=1, propn=0.0, ent=0.0, cap=0.0):
    occ = [Occurrence(sentence=f"a sentence with {lemma} in it", chapter="c", surface=lemma)
           for _ in range(count)]
    c = Candidate(lemma=lemma, pos="NOUN", occurrences=occ,
                  propn_ratio=propn, ent_ratio=ent, cap_ratio=cap)
    c.zipf = zipf if zipf is not None else zipf_frequency(lemma, "en")
    return c


# --- validity gate (§05) -------------------------------------------------

def test_common_headword_kept():
    c = _cand("house")
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.KEEP and "wordlist" in c.validity_sources


def test_misspelling_dropped_even_though_attested():
    # 'definately' has a real web footprint, but a dominant near-twin.
    c = _cand("definately")
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.DROP
    assert c.reject_reason is RejectReason.MISSPELLING


def test_attested_rarity_kept():
    # Not in the 82k wordlist, no dominant twin, but present in the corpus.
    c = _cand("apophenia", zipf=1.6)
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.KEEP and "corpus" in c.validity_sources


def test_archaic_real_word_not_branded_a_misspelling():
    # Regression: real archaic words absent from the 82k wordlist but present in
    # WordNet / the dictionary corpus must not be dropped as misspellings, even
    # when a common near-neighbor exists (destrier ~ 'destroyer').
    gate = ValidityGate(Config())
    for word in ("destrier", "bartizan", "armiger", "rebec"):
        c = _cand(word)
        gate.judge(c)
        assert c.verdict is Verdict.KEEP, f"{word} was wrongly dropped"


def test_recurring_unattested_is_unsure_not_dropped():
    c = _cand("zblargnum", zipf=0.0, count=4)
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.UNSURE  # possible coinage -> review, never silent drop


def test_oneoff_gibberish_dropped():
    c = _cand("xqzptv", zipf=0.0, count=1)
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.DROP and c.reject_reason is RejectReason.NOT_A_WORD


# --- frequency floor (§03.4) --------------------------------------------

def test_floor_drops_common_word():
    cands = {"the": _cand("the")}
    apply_floor(cands, Config())
    assert cands["the"].verdict is Verdict.DROP
    assert cands["the"].reject_reason is RejectReason.FREQUENCY_FLOOR


def test_floor_spares_rare_word():
    cands = {"lugubrious": _cand("lugubrious")}
    apply_floor(cands, Config())
    assert cands["lugubrious"].verdict is None  # survives the floor


# --- proper-noun stripping (§04) ----------------------------------------

def test_oneoff_sentence_initial_propn_is_not_dropped():
    # The 'mote' bug: single PROPN tag on a lone sentence-initial token.
    cands = {"mote": _cand("mote", propn=1.0, count=1, cap=0.0)}
    strip_proper_nouns(cands, Config())
    assert cands["mote"].verdict is None


def test_recurring_propn_is_dropped():
    cands = {"aragorn": _cand("aragorn", propn=1.0, count=3, cap=0.0)}
    strip_proper_nouns(cands, Config())
    assert cands["aragorn"].verdict is Verdict.DROP
    assert cands["aragorn"].reject_reason is RejectReason.PROPER_NOUN


def test_high_capitalization_ratio_is_dropped():
    cands = {"baker": _cand("baker", propn=0.0, count=5, cap=0.9)}
    strip_proper_nouns(cands, Config())
    assert cands["baker"].verdict is Verdict.DROP
