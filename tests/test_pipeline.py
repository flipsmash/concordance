"""pipeline.apply_pruned_exclusions — pure logic, no DB needed."""

from concordance.model import Candidate, RejectReason, Verdict
from concordance.pipeline import apply_pruned_exclusions


def test_pruned_lemma_is_dropped_as_already_known():
    candidates = {"tram": Candidate(lemma="tram", pos="NOUN")}
    n = apply_pruned_exclusions(candidates, {"tram"})
    assert n == 1
    assert candidates["tram"].verdict is Verdict.DROP
    assert candidates["tram"].reject_reason is RejectReason.ALREADY_KNOWN


def test_non_pruned_lemma_is_untouched():
    candidates = {"cangue": Candidate(lemma="cangue", pos="NOUN")}
    n = apply_pruned_exclusions(candidates, {"tram"})
    assert n == 0
    assert candidates["cangue"].verdict is None


def test_does_not_override_an_existing_verdict():
    # Defensive: if some earlier stage already decided this one, don't clobber it.
    c = Candidate(lemma="tram", pos="NOUN")
    c.verdict = Verdict.KEEP
    n = apply_pruned_exclusions({"tram": c}, {"tram"})
    assert n == 0
    assert c.verdict is Verdict.KEEP


def test_empty_pruned_set_excludes_nothing():
    candidates = {"tram": Candidate(lemma="tram", pos="NOUN")}
    n = apply_pruned_exclusions(candidates, set())
    assert n == 0
    assert candidates["tram"].verdict is None
