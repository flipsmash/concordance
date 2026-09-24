"""Context-language classification of book sentences (§ validity)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from concordance import context_lang as cl


def test_target_forms_and_occurrences():
    forms = cl.target_forms("copperware", "Copperwares")
    assert {"copperware", "copperwares", "copperwared"} <= forms
    text = ("He polished the Copperware slowly. Nothing else here.\n\n"
            "The ſchapith line matches ſchapith.")
    got = cl.occurrences(text, {1: forms, 2: {"schapith"}})
    assert got[1] == ["He polished the Copperware slowly."]
    assert len(got[2]) == 2                      # long-s folds to its word, never a KeyError


def test_sentence_at_bounds():
    text = "First one. The target word sits here! Last one."
    i = text.index("target")
    assert cl.sentence_at(text, i, i + 6) == "The target word sits here!"


_MODEL = Path(os.environ.get("CONCORDANCE_CACHE_DIR", Path.home() / ".cache" / "concordance")) / "lid.176.ftz"


@pytest.mark.skipif(not _MODEL.exists(), reason="fastText lid.176 not cached (never downloaded by tests)")
def test_classifier_separates_modern_english_from_medieval_and_foreign():
    pytest.importorskip("fasttext")
    c = cl.ContextClassifier({"shoures", "soote", "droghte", "perced", "swich", "veyne", "licour"})
    assert c.classify("He took the copperware down from the shelf and polished it slowly",
                      {"copperware"})[0] == "english"
    assert c.classify("Er sagte, dass er morgen nach Berlin fahren wolle, aber nicht allein",
                      set()) == ("foreign", "de")
    assert c.classify("Whan that Aprille with his shoures soote the droghte of March hath perced to the roote",
                      set())[0] == "middle_english"
    # Scots dialogue: unknown words but no Middle English markers -> English
    assert c.classify('"Ye ken fine the laird winna thole it," said the auld wife, shaking her heid.',
                      set())[0] == "english"
    assert c.classify("Too short here", set())[0] == "unknown"
