"""pipeline.apply_known_verdicts — the cross-book verdict cache marking.
pipeline._enrich_one — delegates to resolve.resolve_definition(max_tier=MW)
and translates cand.verdict into its own bool return; the MW tier's own
logic (exact-match, homograph pick, foreign-loanword detection) now lives
in resolve.py and is tested there (test_resolve.py), not here."""

from concordance import resolve
from concordance.model import Candidate, RejectReason, Verdict
from concordance.pipeline import _enrich_one, apply_known_verdicts


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


def test_enrich_one_delegates_to_resolve_definition_capped_at_mw(monkeypatch):
    calls = []

    def fake_resolve(cand, **kwargs):
        calls.append(kwargs.get("max_tier"))
        cand.definition = "already resolved"

    monkeypatch.setattr(resolve, "resolve_definition", fake_resolve)
    c = Candidate(lemma="besmirch", pos="VERB")
    result = _enrich_one(c, lexicon={}, oed_lexicon={}, session=None, mw_api_key="fake-key")

    assert result is False
    assert calls == [resolve.Tier.MW]
    assert c.definition == "already resolved"


def test_enrich_one_returns_true_when_resolve_definition_drops_the_candidate(monkeypatch):
    # resolve.Tier.MW's own foreign-loanword detection sets cand.verdict --
    # _enrich_one just needs to surface that as its bool return, whatever
    # set it.
    def fake_resolve(cand, **kwargs):
        cand.definition = "danger"
        cand.verdict = Verdict.DROP
        cand.reject_reason = RejectReason.FOREIGN_LANGUAGE

    monkeypatch.setattr(resolve, "resolve_definition", fake_resolve)
    c = Candidate(lemma="hatari", pos="NOUN")
    result = _enrich_one(c, lexicon={}, oed_lexicon={}, session=None, mw_api_key="fake-key")

    assert result is True
    assert c.verdict is Verdict.DROP
    assert c.reject_reason is RejectReason.FOREIGN_LANGUAGE
    assert c.definition == "danger"
