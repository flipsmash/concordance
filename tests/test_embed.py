"""Semantic-distance embedding — text-fallback resolution is pure logic and
runs always; model-loading tests are separate (see test names below) since
they need real model weights/training data."""

from __future__ import annotations

from concordance import embed


def test_definition_text_prefers_definition():
    assert embed.definition_text("a wooden collar", ["yoke"], "He wore a cangue.") == \
        ("a wooden collar", "definition")


def test_definition_text_falls_back_to_synonyms():
    assert embed.definition_text("", ["yoke", "collar"], "sentence here") == \
        ("yoke, collar", "synonyms")
    assert embed.definition_text(None, ["yoke"], "sentence here") == ("yoke", "synonyms")


def test_definition_text_falls_back_to_sentence():
    assert embed.definition_text("", [], "He wore a cangue.") == ("He wore a cangue.", "sentence")
    assert embed.definition_text(None, None, "He wore a cangue.") == ("He wore a cangue.", "sentence")


def test_definition_text_none_when_nothing_usable():
    assert embed.definition_text("", [], "") is None
    assert embed.definition_text(None, None, None) is None
    assert embed.definition_text("   ", [], "   ") is None


def test_definition_text_skips_blank_synonyms():
    assert embed.definition_text("", ["", "  "], "fallback sentence") == \
        ("fallback sentence", "sentence")
