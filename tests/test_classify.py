"""Classifier code-validation and parsing (no model, no DB)."""

from __future__ import annotations

from concordance import classify


def test_validate_drops_hallucinated_and_repairs_subcodes():
    # G3.1/K5.3 aren't real -> repair to nearest valid ancestor; junk dropped; case fixed
    assert classify._validate(["G3.1", "K5.3", "m4", "junk", "E4.1"]) == ["G3", "K5", "M4"]


def test_validate_caps_at_three_and_dedupes():
    assert classify._validate(["A5.1", "A5.1", "B1", "C1", "E1"]) == ["A5.1", "B1", "C1"]


def test_validate_strips_usas_polarity_markers():
    # USAS marks polarity with +/- ; we ignore it for the code identity
    assert classify._validate(["A5.1+", "E4.1-"]) == ["A5.1", "E4.1"]


def test_validate_ignores_non_list():
    assert classify._validate("A1") == []
    assert classify._validate(None) == []


def test_parse_bare_array():
    assert classify._parse('[{"w":"cannon","c":["G3"]}]') == [{"w": "cannon", "c": ["G3"]}]


def test_parse_strips_fence_and_trailing_prose():
    raw = '```json\n[{"w":"x","c":["A1"]}]\n```\ndone'
    assert classify._parse(raw) == [{"w": "x", "c": ["A1"]}]


def test_parse_garbage_returns_empty():
    assert classify._parse("I cannot comply") == []


def test_prompt_items_injects_wnd_hint(monkeypatch):
    from concordance import wndomains
    monkeypatch.setattr(wndomains, "_lexicon", {"frigate": {"military", "nautical"}})
    items = classify._prompt_items([{"word": "frigate", "definition": "a warship", "sentence": "the frigate sailed"}])
    assert set(items[0]["hint"]) == {"G3", "M4"}     # WND prior surfaced as a hint
