"""Proper-noun stripping (concordance/propernouns.py). Pure — no DB, no model."""

from __future__ import annotations

from concordance.config import Config
from concordance.model import Candidate, Occurrence, RejectReason, Verdict
from concordance.propernouns import strip_proper_nouns


def _c(lemma, *, cap_ratio=0.0, propn_ratio=0.0, ent_ratio=0.0, count=1):
    occs = [Occurrence(sentence="x", chapter="1", surface=lemma) for _ in range(count)]
    return Candidate(lemma=lemma, pos="NOUN", occurrences=occs,
                     cap_ratio=cap_ratio, propn_ratio=propn_ratio, ent_ratio=ent_ratio)


def test_high_cap_ratio_drops_a_genuine_name():
    c = _c("aragorn", cap_ratio=1.0)
    strip_proper_nouns({"aragorn": c}, Config())
    assert c.verdict is Verdict.DROP
    assert c.reject_reason is RejectReason.PROPER_NOUN


def test_high_cap_ratio_still_drops_a_common_word_used_as_a_name():
    # The Bloom/Baker case: cap_ratio is deliberately NOT dictionary-guarded.
    # A common word consistently capitalized throughout one particular book
    # really is being used as a name there. ("merchantability"/"yaks" were
    # wrongly rejected in production, but the actual bug was upstream --
    # ALL-CAPS boilerplate text inflating cap_ratio in tokenize.py's
    # counting, not this module trusting cap_ratio too much; see
    # test_tokenize.py's ALL-CAPS regression tests.)
    c = _c("bloom", cap_ratio=0.98, propn_ratio=0.94, ent_ratio=0.24, count=12)
    strip_proper_nouns({"bloom": c}, Config())
    assert c.verdict is Verdict.DROP
    assert c.reject_reason is RejectReason.PROPER_NOUN


def test_tagger_corroborated_still_drops_a_recurring_mistagged_name():
    c = _c("rochford", propn_ratio=0.8, count=3)
    strip_proper_nouns({"rochford": c}, Config())
    assert c.verdict is Verdict.DROP
    assert c.reject_reason is RejectReason.PROPER_NOUN


def test_tagger_alone_on_a_common_dictionary_word_is_not_dropped():
    # tram/beggar/alderman/sacrament-style tagger misfires: the tagger says
    # PROPN but the word is real, ordinary vocabulary -- the dictionary
    # guard on this branch is untouched by this change and must still work.
    c = _c("tram", propn_ratio=0.9, count=3)
    strip_proper_nouns({"tram": c}, Config())
    assert c.verdict is None


def test_single_occurrence_tagger_hit_is_not_corroborated():
    # A lone sentence-initial PROPN tag can't corroborate itself (count < 2).
    c = _c("motes", propn_ratio=1.0, count=1)
    strip_proper_nouns({"motes": c}, Config())
    assert c.verdict is None


def test_does_not_override_an_existing_verdict():
    c = _c("whatever", cap_ratio=1.0)
    c.verdict = Verdict.KEEP
    strip_proper_nouns({"whatever": c}, Config())
    assert c.verdict is Verdict.KEEP
