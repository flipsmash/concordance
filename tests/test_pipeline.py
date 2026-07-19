"""pipeline.apply_known_verdicts — the cross-book verdict cache marking.
Pure logic, no DB needed."""

from concordance.model import Candidate, RejectReason, Verdict
from concordance.pipeline import apply_known_verdicts


def _cands(*lemmas):
    return {l: Candidate(lemma=l, pos="NOUN") for l in lemmas}


def test_cached_keep_becomes_survivor():
    cands = _cands("cangue")
    counts = apply_known_verdicts(cands, {"cangue": "keep"})
    assert counts["keep"] == 1
    assert cands["cangue"].verdict is Verdict.KEEP
    assert cands["cangue"].reject_reason is None


def test_cached_pruned_is_dropped_already_known():
    cands = _cands("tram")
    counts = apply_known_verdicts(cands, {"tram": "pruned"})
    assert counts["pruned"] == 1
    assert cands["tram"].verdict is Verdict.DROP
    assert cands["tram"].reject_reason is RejectReason.ALREADY_KNOWN


def test_cached_reject_is_dropped_not_interesting():
    cands = _cands("beggar")
    counts = apply_known_verdicts(cands, {"beggar": "not_interesting"})
    assert counts["reject"] == 1
    assert cands["beggar"].verdict is Verdict.DROP
    assert cands["beggar"].reject_reason is RejectReason.NOT_INTERESTING


def test_unknown_lemma_is_untouched():
    cands = _cands("fuligin")
    counts = apply_known_verdicts(cands, {"tram": "pruned"})
    assert counts == {"keep": 0, "pruned": 0, "reject": 0}
    assert cands["fuligin"].verdict is None


def test_does_not_override_an_existing_verdict():
    # Defensive: floor/propernouns may have already decided one.
    c = Candidate(lemma="tram", pos="NOUN")
    c.verdict = Verdict.DROP
    c.reject_reason = RejectReason.FREQUENCY_FLOOR
    counts = apply_known_verdicts({"tram": c}, {"tram": "keep"})
    assert counts["keep"] == 0
    assert c.verdict is Verdict.DROP
    assert c.reject_reason is RejectReason.FREQUENCY_FLOOR


def test_mixed_batch_counts():
    cands = _cands("cangue", "tram", "beggar", "fuligin")
    known = {"cangue": "keep", "tram": "pruned", "beggar": "not_interesting"}
    counts = apply_known_verdicts(cands, known)
    assert counts == {"keep": 1, "pruned": 1, "reject": 1}
    assert cands["fuligin"].verdict is None
