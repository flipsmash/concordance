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


# --- quiz suitability -----------------------------------------------------

def test_quizzable_excludes_variant_forms():
    for defn in ("Plural of goose.", "Obsolete spelling of villain.",
                 "Past participle of write.", "First-person singular of être."):
        ok, reason = quizdef.quizzable(defn)
        assert not ok and reason == "grammatical/variant form"


def test_quizzable_excludes_transparent_derivative_of_common_root():
    # reveller <- revel; 'revel' is common (zipf ~3.6) -> trivially inferable
    ok, reason = quizdef.quizzable("One who revels.", morph_root="revel", root_zipf=3.6)
    assert not ok and "revel" in reason


def test_quizzable_keeps_derivative_of_rare_root():
    # abacination <- abacinate; the root is itself rare -> not inferable, stays quizzable
    ok, reason = quizdef.quizzable("The act of abacinating.", morph_root="abacinate", root_zipf=0.0)
    assert ok and reason == ""


def test_quizzable_keeps_plain_rare_word():
    ok, reason = quizdef.quizzable("A heavy wooden collar borne on the shoulders.")
    assert ok and reason == ""


def test_quizzable_variant_wins_over_missing_root_info():
    # variant check applies even with no morphology supplied
    ok, _ = quizdef.quizzable("Plural of ox.", morph_root=None, root_zipf=None)
    assert not ok


def test_quizzable_handles_empty_definition():
    ok, reason = quizdef.quizzable("")
    assert ok and reason == ""


def test_quizzable_no_exclusion_when_root_below_threshold():
    # root present but uncommon (zipf < 3.0) -> still quizzable
    ok, reason = quizdef.quizzable("The act of gnarring.", morph_root="gnar", root_zipf=1.2)
    assert ok and reason == ""
