"""Unified definition-acquisition cascade (concordance/resolve.py). Every
tier function is monkeypatched -- this tests cascade ordering, max_tier
cutoff, and POS-repair, not the tier functions' own network/parsing logic
(those are covered by test_deepdef.py/test_dictionary.py's own tests)."""

from __future__ import annotations

import pytest

from concordance import resolve
from concordance.model import Candidate


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(resolve.time, "sleep", lambda *a, **k: None)


def _cand(lemma="testword"):
    return Candidate(lemma=lemma, pos="NOUN")


def _miss(*a, **k):
    return False


def test_local_hit_stops_the_cascade(monkeypatch):
    monkeypatch.setattr(resolve.localdict, "enrich", lambda cand, lex: (
        setattr(cand, "definition", "a local def") or setattr(cand, "part_of_speech", "noun") or True
    ))
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda *a, **k: pytest.fail("FREE tier should not run"))
    c = _cand()
    assert resolve.resolve_definition(c, lexicon={"testword": [("noun", "x", "", "", False, False)]}) is resolve.Tier.LOCAL
    assert c.definition == "a local def"


def test_falls_through_to_free_tier_on_local_miss(monkeypatch):
    monkeypatch.setattr(resolve.localdict, "enrich", _miss)

    def fake_freedict(cand, session):
        cand.definition = "a free def"
        cand.definition_source = "Free Dictionary API"

    monkeypatch.setattr(resolve.dictionary, "enrich", fake_freedict)
    c = _cand()
    assert resolve.resolve_definition(c, session=object()) is resolve.Tier.FREE
    assert c.definition == "a free def"


def test_max_tier_cutoff_stops_before_wordnik(monkeypatch):
    monkeypatch.setattr(resolve.localdict, "enrich", _miss)
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)  # leaves definition blank
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "KEY")
    monkeypatch.setattr(resolve.deepdef, "_from_wordnik", lambda *a, **k: pytest.fail("WORDNIK should not run"))
    c = _cand()
    assert resolve.resolve_definition(c, max_tier=resolve.Tier.FREE, session=object()) is None


def test_wordnik_skipped_without_a_key(monkeypatch):
    monkeypatch.setattr(resolve.localdict, "enrich", _miss)
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
    called = {"wordnik": False}
    monkeypatch.setattr(resolve.deepdef, "_from_wordnik", lambda *a, **k: called.__setitem__("wordnik", True) or True)
    monkeypatch.setattr(resolve.deepdef, "_from_yourdictionary", _miss)
    c = _cand()
    resolve.resolve_definition(c, session=object())
    assert called["wordnik"] is False


def test_falls_through_to_yourdictionary_when_wordnik_misses(monkeypatch):
    monkeypatch.setattr(resolve.localdict, "enrich", _miss)
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "KEY")
    monkeypatch.setattr(resolve.deepdef, "_from_wordnik", _miss)

    def fake_yd(cand, session):
        cand.definition = "a yourdictionary def"
        cand.definition_source = "yourdictionary.com"
        return True

    monkeypatch.setattr(resolve.deepdef, "_from_yourdictionary", fake_yd)
    c = _cand()
    assert resolve.resolve_definition(c, session=object()) is resolve.Tier.YOURDICT


def test_web_tier_only_runs_with_an_llm(monkeypatch):
    monkeypatch.setattr(resolve.localdict, "enrich", _miss)
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(resolve.deepdef, "_from_yourdictionary", _miss)
    monkeypatch.setattr(resolve.websearch, "define_via_web", lambda *a, **k: pytest.fail("WEB should not run"))
    c = _cand()
    assert resolve.resolve_definition(c, session=object(), llm=None) is None


def test_web_tier_resolves_with_an_llm(monkeypatch):
    monkeypatch.setattr(resolve.localdict, "enrich", _miss)
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(resolve.deepdef, "_from_yourdictionary", _miss)

    def fake_web(cand, llm):
        cand.definition = "a web def"
        cand.definition_source = "Web (LLM-extracted)"
        return True

    monkeypatch.setattr(resolve.websearch, "define_via_web", fake_web)
    c = _cand()
    assert resolve.resolve_definition(c, session=object(), llm=object()) is resolve.Tier.WEB


def test_nothing_resolves_returns_none(monkeypatch):
    monkeypatch.setattr(resolve.localdict, "enrich", _miss)
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(resolve.deepdef, "_from_yourdictionary", _miss)
    c = _cand()
    assert resolve.resolve_definition(c, session=object(), llm=None) is None
    assert c.definition == ""


def test_pos_repair_borrows_from_lexicon_without_touching_definition(monkeypatch):
    # The web tier never sets part_of_speech (a real, structural gap) --
    # confirm resolve_definition backfills it from the lexicon afterward,
    # without disturbing the definition/source the web tier actually won.
    monkeypatch.setattr(resolve.localdict, "enrich", _miss)
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(resolve.deepdef, "_from_yourdictionary", _miss)

    def fake_web(cand, llm):
        cand.definition = "a web def"
        cand.definition_source = "Web (LLM-extracted)"
        return True

    monkeypatch.setattr(resolve.websearch, "define_via_web", fake_web)
    c = _cand("borrowme")
    lexicon = {"borrowme": [("verb", "some local gloss", "", "", False, False)]}
    result = resolve.resolve_definition(c, lexicon=lexicon, session=object(), llm=object())
    assert result is resolve.Tier.WEB
    assert c.definition == "a web def"                # unchanged -- web tier's text wins
    assert c.definition_source == "Web (LLM-extracted)"
    assert c.part_of_speech == "verb"                  # borrowed from the lexicon


def test_pos_repair_is_a_noop_when_lexicon_has_no_entry(monkeypatch):
    # A hit that (unusually) leaves POS blank, with nothing in the lexicon
    # for this lemma to borrow from -- must stay blank, not error.
    def fake_local(cand, lex):
        cand.definition = "already resolved elsewhere"
        return True

    monkeypatch.setattr(resolve.localdict, "enrich", fake_local)
    c = _cand("unknownword")
    result = resolve.resolve_definition(c, lexicon={}, session=object())
    assert result is resolve.Tier.LOCAL
    assert c.part_of_speech == ""  # nothing to borrow -- stays blank, not an error
