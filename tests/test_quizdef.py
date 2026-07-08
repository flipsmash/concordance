"""Definition leak detection + redaction (pure; LLM rewrite exercised live)."""
from __future__ import annotations
from concordance import quizdef


def test_detects_manner_adverb_leak():
    assert "audacious" in quizdef.leaking_tokens("audaciously", "In an audacious manner.")


def test_detects_agent_and_quality_leaks():
    assert quizdef.has_leak("borderer", "A person who resides near a border.")
    assert quizdef.has_leak("baseness", "The quality of being base.")
    assert quizdef.has_leak("congealment", "The act of congealing.")


def test_detects_stem_variant_leak():
    # different surface, same stem
    assert quizdef.has_leak("pleader", "a person who pleads in court")


def test_clean_definition_has_no_leak():
    assert not quizdef.has_leak("cangue", "A heavy wooden collar borne on the shoulders.")


def test_unrelated_short_prefix_is_not_a_leak():
    # 'cat' vs 'catalogue' share 3 chars only -> not flagged
    assert not quizdef._shared_root("cat", "catalogue")


def test_redact_blanks_leaking_tokens_only():
    out = quizdef.redact("borderer", "A person who resides near a border.")
    assert "border" not in out.lower() and "person" in out
