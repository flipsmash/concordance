"""Tests for the judge's parsing and verdict application — the parts that must
be correct regardless of what the (nondeterministic) model emits."""

from concordance.judge import (
    _freq_band,
    _parse_verdicts,
    _verdict_keep,
    _verdict_word,
    apply_verdicts,
)
from concordance.model import Candidate, RejectReason, Verdict


# --- frequency band hint --------------------------------------------------

def test_freq_band_thresholds():
    assert _freq_band(4.0) == "common"
    assert _freq_band(3.0) == "common"      # boundary
    assert _freq_band(2.5) == "uncommon"
    assert _freq_band(2.0) == "uncommon"    # boundary
    assert _freq_band(1.2) == "rare"
    assert _freq_band(0.0) == "rare"


def _c(lemma, verdict=Verdict.KEEP):
    c = Candidate(lemma=lemma, pos="NOUN")
    c.verdict = verdict
    return c


# --- parsing --------------------------------------------------------------

def test_parse_bare_array():
    assert _parse_verdicts('[{"word":"x","keep":true,"reason":"r"}]') == [
        {"word": "x", "keep": True, "reason": "r"}
    ]


def test_parse_object_wrapper():
    out = _parse_verdicts('{"words":[{"word":"x","keep":false,"reason":"r"}]}')
    assert out == [{"word": "x", "keep": False, "reason": "r"}]


def test_parse_strips_code_fence_and_trailing_prose():
    raw = '```json\n[{"word":"x","keep":true,"reason":"r"}]\n```\nHope that helps!'
    assert _parse_verdicts(raw) == [{"word": "x", "keep": True, "reason": "r"}]


def test_parse_garbage_returns_none():
    assert _parse_verdicts("I could not comply.") is None


# --- application ----------------------------------------------------------

def test_keep_false_drops_a_kept_word():
    c = _c("marbled")
    apply_verdicts([c], [{"word": "marbled", "keep": False, "reason": "everyday"}])
    assert c.verdict is Verdict.DROP
    assert c.reject_reason is RejectReason.NOT_INTERESTING


def test_keep_true_records_reason_and_survives():
    c = _c("susurrus")
    apply_verdicts([c], [{"word": "susurrus", "keep": True, "reason": "vivid, literary"}])
    assert c.verdict is Verdict.KEEP
    assert c.interesting_reason == "vivid, literary"


def test_unsure_is_dropped_by_a_negative_judge_verdict():
    # UNSURE means "send to review," but `ingest` has no human review step --
    # the judge itself is the review, so its keep:false must actually apply
    # here too, not just to a plain KEEP. (Previously this silently never
    # happened: UNSURE was veto-proof, so every misspelling/foreign-word
    # candidate that reached UNSURE was permanently kept regardless of what
    # the judge said.)
    c = _c("zblargnum", verdict=Verdict.UNSURE)
    apply_verdicts([c], [{"word": "zblargnum", "keep": False, "reason": "nonsense"}])
    assert c.verdict is Verdict.DROP
    assert c.reject_reason is RejectReason.NOT_INTERESTING


def test_unsure_survives_a_positive_judge_verdict():
    c = _c("necropoli", verdict=Verdict.UNSURE)
    apply_verdicts([c], [{"word": "necropoli", "keep": True, "reason": "recurring coinage"}])
    assert c.verdict is Verdict.UNSURE
    assert c.interesting_reason == "recurring coinage"


def test_word_missing_from_verdicts_is_kept():
    c = _c("orphan")
    apply_verdicts([c], [{"word": "other", "keep": False, "reason": "x"}])
    assert c.verdict is Verdict.KEEP  # keep-biased: no verdict => keep


def test_unparseable_batch_keeps_all():
    c = _c("safe")
    apply_verdicts([c], None)
    assert c.verdict is Verdict.KEEP
    assert "parse error" in c.interesting_reason


def test_case_insensitive_word_match():
    c = _c("brittle")
    apply_verdicts([c], [{"word": "Brittle", "keep": False, "reason": "everyday"}])
    assert c.verdict is Verdict.DROP


# --- compact {"w","k"} schema (the production format) ---------------------

def test_compact_keep_false_drops():
    c = _c("stink")
    apply_verdicts([c], [{"w": "stink", "k": False}])
    assert c.verdict is Verdict.DROP
    assert c.reject_reason is RejectReason.NOT_INTERESTING


def test_compact_keep_true_survives():
    c = _c("cangue")
    apply_verdicts([c], [{"w": "cangue", "k": True}])
    assert c.verdict is Verdict.KEEP


def test_verdict_helpers_read_both_schemas():
    assert _verdict_word({"w": "Foo"}) == "foo"
    assert _verdict_word({"word": "Bar"}) == "bar"
    assert _verdict_word("not a dict") == ""
    assert _verdict_keep({"k": False}) is False
    assert _verdict_keep({"keep": False}) is False
    assert _verdict_keep({}) is True          # keep-biased default


def test_mixed_batch_drops_commons_keeps_rares():
    common = _c("whisper")
    rare = _c("refectory")
    apply_verdicts([common, rare], [{"w": "whisper", "k": False}, {"w": "refectory", "k": True}])
    assert common.verdict is Verdict.DROP
    assert rare.verdict is Verdict.KEEP
