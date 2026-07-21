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


def test_detects_prefix_derived_leak():
    # "premeditate" = "pre" + "meditate" -- shares NO leading characters with
    # its base at all, so the front-alignment check alone can never catch
    # this; needs the explicit prefix-list check.
    assert quizdef.has_leak("premeditate", "To meditate, consider, or plan beforehand")
    assert "meditate" in quizdef.leaking_tokens("premeditate", "To meditate, consider, or plan beforehand")
    assert quizdef.has_leak("irremovable", "Not removable")
    assert "removable" in quizdef.leaking_tokens("irremovable", "Not removable")


def test_prefix_check_does_not_false_positive_on_coincidental_endings():
    # Both happen to end in "-ation" but are unrelated words -- a generic
    # shared-suffix scan would wrongly flag this; the targeted prefix-list
    # check must not.
    assert not quizdef._shared_root("nation", "creation")


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


# --- redaction sparsity ------------------------------------------------------

def test_content_word_count_ignores_function_words():
    assert quizdef._content_word_count("A male dealer in —.") == 2  # male, dealer
    assert quizdef._content_word_count("—") == 0
    # pertaining, boxing, fighting
    assert quizdef._content_word_count("Of or pertaining to boxing or fighting with —.") == 3


def test_redaction_too_sparse_flags_near_empty_definitions():
    # Regression: "silkman" -> "A male dealer in silk." redacted to "A male
    # dealer in —." is grammatically intact but no longer distinguishes
    # silkman from any other "male dealer in X" trade word.
    assert quizdef.redaction_too_sparse("A male dealer in —.")
    assert quizdef.redaction_too_sparse("—")
    assert quizdef.redaction_too_sparse("One who —.")


def test_redaction_too_sparse_keeps_content_rich_redaction():
    # A definition with enough surviving content still functions as a clue
    # even with one term blanked.
    assert not quizdef.redaction_too_sparse(
        "A device for holding one or more lit — near a window during a storm.")


def test_quizzable_excludes_over_redacted_definition():
    ok, reason = quizdef.quizzable(
        "A male dealer in silk.", quiz_definition="A male dealer in —.", quiz_def_source="redacted")
    assert not ok
    assert "redaction" in reason


def test_quizzable_ignores_sparsity_check_for_non_redacted_sources():
    # A short CLEAN or REWRITTEN definition was never blanked -- brevity
    # alone isn't grounds for exclusion, only redaction that hollowed it out.
    ok, reason = quizdef.quizzable(
        "Wealth, riches.", quiz_definition="Wealth, riches.", quiz_def_source="clean")
    assert ok and reason == ""


def test_rewrite_chunks_the_hard_retry_pass():
    # Regression: rewrite() chunked the normal pass by self.batch but passed
    # the ENTIRE too-sparse subset to _batch_hard in one call -- on a real
    # backlog (1,407 too-sparse items) that built a single prompt payload
    # requesting far more tokens than the model's context window, and
    # crashed with "Requested tokens (22264) exceed context window of 8192."
    from unittest.mock import MagicMock, patch

    rw = quizdef.Rewriter.__new__(quizdef.Rewriter)
    rw.llm = MagicMock()
    rw.batch = 10

    # Every item here is short enough that its redaction is too sparse, so
    # all of them fall into the would_redact_sparse / hard-retry path.
    items = [{"word": f"tradesman{i}", "definition": f"A dealer in tradesman{i}."} for i in range(25)]

    call_sizes = []

    def fake_query_hard(pending):
        call_sizes.append(len(pending))
        return []  # model returns nothing -> everything falls to redact()

    with patch.object(rw, "_query", return_value=[]), patch.object(rw, "_query_hard", side_effect=fake_query_hard):
        rw.rewrite(items)

    assert call_sizes, "the hard-retry path was never exercised"
    assert max(call_sizes) <= rw.batch, f"a hard-retry batch exceeded self.batch: {call_sizes}"
