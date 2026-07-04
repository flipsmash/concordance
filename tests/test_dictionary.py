"""Enrichment robustness + parsing (no network — sessions are faked)."""

from __future__ import annotations

import pytest

from concordance import dictionary as D
from concordance.model import Candidate


class _Resp:
    def __init__(self, code, headers=None, payload=None):
        self.status_code = code
        self.headers = headers or {}
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Session:
    """Serves a queued list of responses/exceptions, one per get()."""

    def __init__(self, seq):
        self.seq = list(seq)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        item = self.seq[min(self.calls, len(self.seq) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda *a, **k: None)


def test_get_retries_then_succeeds():
    s = _Session([_Resp(429, {"Retry-After": "1"}), _Resp(503), _Resp(200)])
    r = D._get(s, "http://x")
    assert r.status_code == 200
    assert s.calls == 3


def test_get_gives_up_after_max_tries():
    s = _Session([_Resp(429)])
    r = D._get(s, "http://x")
    assert r.status_code == 429
    assert s.calls == D._MAX_TRIES


def test_get_returns_none_on_persistent_exception():
    import requests
    s = _Session([requests.RequestException("boom")])
    assert D._get(s, "http://x") is None
    assert s.calls == D._MAX_TRIES


def test_get_passes_through_non_retryable():
    s = _Session([_Resp(404)])
    r = D._get(s, "http://x")
    assert r.status_code == 404
    assert s.calls == 1


def test_parse_etymology_stops_at_next_header():
    text = (
        "== English ==\n\n=== Etymology ===\n"
        "From Latin foo, from Ancient Greek bar.\n\n"
        "=== Noun ===\nfoo (plural foos)\nA thing."
    )
    ety = D._parse_etymology(text)
    assert "Latin foo" in ety
    assert "Noun" not in ety and "plural" not in ety


def test_parse_etymology_absent():
    assert D._parse_etymology("== English ==\n=== Noun ===\nA thing.") == ""


def test_parse_ipa():
    assert D._parse_ipa("=== Pronunciation ===\nIPA: /kæŋ/\n") == "/kæŋ/"
    assert D._parse_ipa("no pronunciation here") == ""


def test_enrich_falls_back_to_wiktionary(monkeypatch):
    """Freedict 404 -> Wiktionary supplies the definition; source is set."""
    monkeypatch.setattr(D, "_from_freedict", lambda c, s: False)

    def fake_wikt(c, s):
        c.definition = "a real word"
        return True

    monkeypatch.setattr(D, "_from_wiktionary", fake_wikt)
    monkeypatch.setattr(D, "_augment_from_raw", lambda c, s: None)
    cand = Candidate(lemma="cangue", pos="NOUN")
    D.enrich(cand, session=object())
    assert cand.definition == "a real word"
    assert cand.definition_source == "Wiktionary"


def test_enrich_no_source_stays_empty(monkeypatch):
    monkeypatch.setattr(D, "_from_freedict", lambda c, s: False)
    monkeypatch.setattr(D, "_from_wiktionary", lambda c, s: False)
    cand = Candidate(lemma="zzzz", pos="NOUN")
    D.enrich(cand, session=object())
    assert cand.definition == ""
    assert cand.definition_source == ""
