"""Book/author fame scoring — network and LLM faked throughout."""

from __future__ import annotations

from concordance import fame


class _Resp:
    def __init__(self, code=200, payload=None):
        self.status_code = code
        self.headers = {}   # dictionary._get's retry path reads Retry-After off this
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError
        return self._payload


class _WikidataSession:
    """Fakes the two sequential Wikidata calls wikidata_lookup makes,
    branching on the `action` param the same way test_ngram.py's _Session
    branches on nothing (there's only ever one call there)."""

    def __init__(self, search_payload, get_payload=None):
        self.search_payload = search_payload
        self.get_payload = get_payload

    def get(self, url, params=None, timeout=None):
        if params.get("action") == "wbsearchentities":
            return _Resp(payload=self.search_payload)
        return _Resp(payload=self.get_payload)


class _LLM:
    def __init__(self, reply):
        self.reply = reply
        self.seen = None

    def create_chat_completion(self, messages, temperature=0.0, max_tokens=200):
        self.seen = messages
        return {"choices": [{"message": {"content": self.reply}}]}


# --- _natural_name -----------------------------------------------------------

def test_natural_name_reorders_last_first():
    assert fame._natural_name("Galsworthy, John") == "John Galsworthy"


def test_natural_name_prefers_parenthetical_full_expansion():
    # Confirmed live: this is the dominant real format in the corpus
    # (3,883/3,992 distinct authors, 97%) -- the parenthetical spells out
    # abbreviated initials in full, which searches far better than the
    # abbreviated form.
    assert fame._natural_name("Jacobs, W. W. (William Wymark)") == "William Wymark Jacobs"
    assert fame._natural_name("Stacpoole, H. De Vere (Henry De Vere)") == "Henry De Vere Stacpoole"


def test_natural_name_no_comma_passes_through():
    assert fame._natural_name("Various") == "Various"
    assert fame._natural_name("Madonna") == "Madonna"


def test_natural_name_empty_given_name_falls_back_to_original():
    assert fame._natural_name("Smith,") == "Smith,"


# --- _parse_fame_verdict / _score_from_verdict ------------------------------

def test_parse_fame_verdict_plain_json():
    assert fame._parse_fame_verdict('{"score": 7, "why": "well known"}') == {"score": 7, "why": "well known"}


def test_parse_fame_verdict_code_fenced():
    text = '```json\n{"score": 3, "why": "obscure"}\n```'
    assert fame._parse_fame_verdict(text) == {"score": 3, "why": "obscure"}


def test_parse_fame_verdict_prose_wrapped():
    text = 'Sure, here is my answer: {"score": 9, "why": "canonical"} Hope that helps!'
    assert fame._parse_fame_verdict(text)["score"] == 9


def test_parse_fame_verdict_unparseable_returns_none():
    assert fame._parse_fame_verdict("I cannot answer that.") is None


def test_score_from_verdict_valid():
    assert fame._score_from_verdict({"score": 6, "why": "ok"}) == (6.0, "ok")


def test_score_from_verdict_out_of_range_is_a_failure_not_a_clamp():
    assert fame._score_from_verdict({"score": 11, "why": "bad"}) == (None, "")
    assert fame._score_from_verdict({"score": 0, "why": "bad"}) == (None, "")
    assert fame._score_from_verdict({"score": -3, "why": "bad"}) == (None, "")


def test_score_from_verdict_non_numeric_is_a_failure():
    assert fame._score_from_verdict({"score": "very famous", "why": "bad"}) == (None, "")


def test_score_from_verdict_missing_verdict():
    assert fame._score_from_verdict(None) == (None, "")


def test_score_from_verdict_why_defaults_to_empty_not_crash():
    assert fame._score_from_verdict({"score": 5}) == (5.0, "")


# --- wikidata_lookup ---------------------------------------------------------

def test_wikidata_lookup_corroborated_match():
    session = _WikidataSession(
        search_payload={"search": [{"id": "Q692", "label": "William Shakespeare"}]},
        get_payload={"entities": {"Q692": {
            "sitelinks": {f"lang{i}wiki": {} for i in range(335)},
            "descriptions": {"en": {"value": "English playwright and poet"}},
        }}},
    )
    result = fame.wikidata_lookup("William Shakespeare", session)
    assert result["qid"] == "Q692"
    assert result["sitelinks"] == 335
    assert result["corroborated"] is True


def test_wikidata_lookup_uncorroborated_match_flagged():
    # Regression: wbsearchentities' fuzzy match can resolve to the wrong
    # entity for a common name (confirmed live: "Rex Beach" matched an
    # unrelated 0-sitelink disambiguation stub) -- a description that
    # doesn't read as a writer must be flagged, not trusted at face value.
    session = _WikidataSession(
        search_payload={"search": [{"id": "Q999", "label": "Rex Beach (community)"}]},
        get_payload={"entities": {"Q999": {
            "sitelinks": {},
            "descriptions": {"en": {"value": "unincorporated community in Florida"}},
        }}},
    )
    result = fame.wikidata_lookup("Rex Beach", session)
    assert result["corroborated"] is False


def test_wikidata_lookup_no_search_hits_returns_none():
    session = _WikidataSession(search_payload={"search": []})
    assert fame.wikidata_lookup("zzzznotarealname", session) is None


def test_wikidata_lookup_bad_response_returns_none():
    class _DeadSession:
        def get(self, url, params=None, timeout=None):
            return _Resp(code=404)   # terminal, non-retryable -- keeps the test fast

    assert fame.wikidata_lookup("anyone", _DeadSession()) is None


# --- gather_author_evidence / gather_book_evidence --------------------------

def test_gather_author_evidence_marks_every_failure_explicitly(monkeypatch):
    # Every dependency fails -- the returned factors must still name each
    # one as failed, never silently omit a key (a silent gap would look
    # identical to "evidence considered and found weak").
    monkeypatch.setattr(fame, "ngram_fetch", lambda name, session: None)
    monkeypatch.setattr(fame, "wikidata_lookup", lambda name, session: None)
    monkeypatch.setattr(fame, "search_snippets", lambda query, max_results=6: [])

    factors = fame.gather_author_evidence("Nobody Famous", session=object())
    assert factors["ngram"]["failed"] is True
    assert factors["wikidata"]["failed"] is True
    assert factors["snippets"] == []
    assert factors["snippets_failed"] is True


def test_gather_author_evidence_records_real_signals(monkeypatch):
    monkeypatch.setattr(fame, "ngram_fetch", lambda name, session: {"peak": 7e-6, "recent": 3e-6, "recency_ratio": 0.4, "peak_year": 1600})
    monkeypatch.setattr(fame, "wikidata_lookup", lambda name, session: {"qid": "Q692", "sitelinks": 335, "description": "playwright", "corroborated": True})
    monkeypatch.setattr(fame, "search_snippets", lambda query, max_results=6: ["Shakespeare was an English playwright..."])

    factors = fame.gather_author_evidence("William Shakespeare", session=object())
    assert factors["ngram"]["peak"] == 7e-6
    assert factors["wikidata"]["sitelinks"] == 335
    assert factors["snippets_failed"] is False


def test_gather_book_evidence_skips_ngram_for_long_title(monkeypatch):
    calls = []
    monkeypatch.setattr(fame, "ngram_fetch", lambda title, session: calls.append(title) or None)
    monkeypatch.setattr(fame, "search_snippets", lambda query, max_results=6: [])

    long_title = "The Extremely Long and Overwrought Title of an Obscure Nineteenth Century Novel"
    factors = fame.gather_book_evidence(long_title, "Some Author", None, session=object())
    assert calls == []   # ngram_fetch never called
    assert factors["ngram"]["skipped"] is True


def test_gather_book_evidence_queries_ngram_for_short_title(monkeypatch):
    calls = []
    monkeypatch.setattr(fame, "ngram_fetch", lambda title, session: calls.append(title) or {"peak": 1e-5, "recent": 1e-5, "recency_ratio": 1.0, "peak_year": 1600})
    monkeypatch.setattr(fame, "search_snippets", lambda query, max_results=6: [])

    factors = fame.gather_book_evidence("Hamlet", "William Shakespeare", None, session=object())
    assert calls == ["Hamlet"]
    assert factors["ngram"]["failed"] is False


def test_gather_book_evidence_records_author_fame_context_verbatim(monkeypatch):
    monkeypatch.setattr(fame, "ngram_fetch", lambda title, session: None)
    monkeypatch.setattr(fame, "search_snippets", lambda query, max_results=6: [])

    author_fame = {"fame_score": 10.0, "fame_reasoning": "foundational figure", "computed_at": "2026-07-29"}
    factors = fame.gather_book_evidence("Hamlet", "William Shakespeare", author_fame, session=object())
    assert factors["author_fame_seen"] == author_fame


def test_gather_book_evidence_author_fame_none_recorded_as_none(monkeypatch):
    monkeypatch.setattr(fame, "ngram_fetch", lambda title, session: None)
    monkeypatch.setattr(fame, "search_snippets", lambda query, max_results=6: [])

    factors = fame.gather_book_evidence("Some Obscure Book", "Some Obscure Author", None, session=object())
    assert factors["author_fame_seen"] is None


# --- score_author / score_book -----------------------------------------------

def test_score_author_happy_path():
    llm = _LLM('{"score": 10, "why": "foundational, universally known"}')
    score, why = fame.score_author(llm, "William Shakespeare", {"ngram": {}, "wikidata": {}})
    assert score == 10.0
    assert why == "foundational, universally known"


def test_score_author_retries_once_on_unparseable_then_succeeds(monkeypatch):
    replies = iter(["not json at all", '{"score": 2, "why": "obscure"}'])
    monkeypatch.setattr(fame, "_query_llm", lambda llm, system, user: next(replies))
    score, why = fame.score_author(object(), "Someone Obscure", {})
    assert score == 2.0


def test_score_author_gives_up_after_max_tries(monkeypatch):
    monkeypatch.setattr(fame, "_query_llm", lambda llm, system, user: "still not json")
    score, why = fame.score_author(object(), "Someone Obscure", {}, tries=2)
    assert score is None
    assert why == ""


def test_score_book_with_no_author_fame_context_does_not_crash():
    llm = _LLM('{"score": 3, "why": "unremarkable"}')
    score, why = fame.score_book(llm, "Some Obscure Book", "Some Obscure Author",
                                  {"ngram": {"skipped": True}, "author_fame_seen": None})
    assert score == 3.0


def test_score_book_rubric_mentions_author_score_is_context_not_a_floor():
    # Not a behavioral test of the LLM (there is none here) -- a guard that
    # the rubric text itself still contains the explicit "don't inflate
    # toward the author's score" instruction, since a bare "use as context"
    # instruction is exactly the kind of soft constraint a mid-size local
    # model ignores without a concrete counter-example (judge.py's own
    # documented lesson).
    assert "NOT inflated toward the author's score" in fame.BOOK_RUBRIC
