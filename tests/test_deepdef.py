"""Deep definition sources (Wordnik / yourdictionary) — network faked."""

from __future__ import annotations

import pytest

from concordance import deepdef
from concordance.model import Candidate


class _Resp:
    def __init__(self, code=200, payload=None, text=""):
        self.status_code = code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Session:
    def __init__(self, resp):
        self.resp = resp

    def get(self, url, params=None, timeout=None):
        return self.resp


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(deepdef.time if hasattr(deepdef, "time") else __import__("time"),
                        "sleep", lambda *a, **k: None, raising=False)


def test_wordnik_prefers_century_and_sets_source():
    payload = [
        {"text": "modern gloss", "sourceDictionary": "ahd-5", "partOfSpeech": "noun"},
        {"text": "archaic gloss", "sourceDictionary": "century", "partOfSpeech": "noun"},
    ]
    c = Candidate(lemma="scrimer", pos="NOUN")
    assert deepdef._from_wordnik(c, _Session(_Resp(payload=payload)), "KEY") is True
    assert c.definition == "archaic gloss"                 # century wins
    assert c.definition_source == "Wordnik (century)"


def test_wordnik_prefers_a_posed_entry_over_a_same_rank_crossreference():
    # Regression: real Wordnik data for "cangue" -- two century entries, the
    # API returns the no-POS cross-reference gloss FIRST. A stable sort on
    # source-dictionary rank alone leaves it in front of the actual noun
    # definition; the tiebreaker must prefer the POS-carrying entry.
    payload = [
        {"text": "To sentence to the cangue.", "sourceDictionary": "century"},  # no partOfSpeech
        {"text": "A heavy wooden collar worn as punishment.",
         "sourceDictionary": "century", "partOfSpeech": "noun"},
    ]
    c = Candidate(lemma="cangue", pos="NOUN")
    assert deepdef._from_wordnik(c, _Session(_Resp(payload=payload)), "KEY") is True
    assert c.definition == "A heavy wooden collar worn as punishment."
    assert c.part_of_speech == "noun"


def test_wordnik_empty_text_is_a_miss():
    c = Candidate(lemma="cobloaf", pos="NOUN")
    payload = [{"text": "", "sourceDictionary": "century"}]
    assert deepdef._from_wordnik(c, _Session(_Resp(payload=payload)), "KEY") is False
    assert c.definition == ""


def test_yourdictionary_extracts_meta_gloss():
    html = '<meta name="description" content="Ungenitured definition: (obsolete) Destitute of genitals; impotent.">'
    c = Candidate(lemma="ungenitured", pos="ADJ")
    assert deepdef._from_yourdictionary(c, _Session(_Resp(text=html))) is True
    assert "Destitute of genitals" in c.definition
    assert c.definition_source == "yourdictionary.com"


def test_yourdictionary_notfound_blurb_is_a_miss():
    html = '<meta name="description" content="Find your word on YourDictionary today!">'
    c = Candidate(lemma="zxqwplt", pos="NOUN")
    assert deepdef._from_yourdictionary(c, _Session(_Resp(text=html))) is False


def test_deep_enrich_cascade_falls_through(monkeypatch):
    monkeypatch.setattr(deepdef, "_from_wordnik", lambda c, s, k: False)
    monkeypatch.setattr(deepdef, "_from_yourdictionary", lambda c, s: False)
    c = Candidate(lemma="prabble", pos="NOUN")
    assert deepdef.deep_enrich(c, _Session(_Resp()), key="KEY") is False


def test_deep_enrich_skips_wordnik_without_key(monkeypatch):
    called = {"wordnik": False}
    def wk(c, s, k):
        called["wordnik"] = True
        return False
    monkeypatch.setattr(deepdef, "_from_wordnik", wk)
    monkeypatch.setattr(deepdef, "_from_yourdictionary", lambda c, s: False)
    deepdef.deep_enrich(Candidate(lemma="x", pos="NOUN"), _Session(_Resp()), key="")
    assert called["wordnik"] is False                      # no key => Wordnik skipped
