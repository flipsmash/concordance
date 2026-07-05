"""Web-search + grounded extraction — LLM and network faked."""

from __future__ import annotations

from concordance import websearch
from concordance.model import Candidate


class _LLM:
    def __init__(self, reply):
        self.reply = reply
        self.seen = None

    def create_chat_completion(self, messages, temperature=0.0, max_tokens=64):
        self.seen = messages
        return {"choices": [{"message": {"content": self.reply}}]}


def test_extract_returns_gloss_from_snippets():
    llm = _LLM("A crusty uneven loaf with a round top.")
    got = websearch.extract_definition("cobloaf", ["Cobloaf — a crusty loaf..."], llm)
    assert got == "A crusty uneven loaf with a round top."


def test_extract_none_when_model_says_none():
    llm = _LLM("NONE")
    assert websearch.extract_definition("prabble", ["unrelated text"], llm) == ""


def test_extract_empty_snippets_skips_model():
    llm = _LLM("should not be used")
    assert websearch.extract_definition("x", [], llm) == ""
    assert llm.seen is None                       # model never called


def test_extract_rejects_refusal_prefixes():
    for bad in ["I could not find a definition.", "The snippets do not define it.", "Sorry, none."]:
        assert websearch.extract_definition("x", ["some text"], _LLM(bad)) == ""


def test_define_via_web_sets_source(monkeypatch):
    monkeypatch.setattr(websearch, "search_snippets", lambda w, max_results=6: ["real snippet"])
    c = Candidate(lemma="cobloaf", pos="NOUN")
    assert websearch.define_via_web(c, _LLM("a round hollow loaf")) is True
    assert c.definition == "a round hollow loaf"
    assert c.definition_source == "Web (LLM-extracted)"


def test_define_via_web_miss_leaves_candidate(monkeypatch):
    monkeypatch.setattr(websearch, "search_snippets", lambda w, max_results=6: [])
    c = Candidate(lemma="zxqwplt", pos="NOUN")
    assert websearch.define_via_web(c, _LLM("NONE")) is False
    assert c.definition == ""
