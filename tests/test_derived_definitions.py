"""Derived definitions for undefined words with a known root (no model)."""

from __future__ import annotations

from concordance import derived_definitions as dd


class _Cur:
    def __init__(self, words, wikt):
        self.words, self.wikt, self.q = words, wikt, ""

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def execute(self, sql, params=None):
        self.q = sql

    def fetchall(self):
        return self.wikt if "wiktionary" in self.q else self.words


class _Conn:
    def __init__(self, words, wikt):
        self.c = _Cur(words, wikt)

    def cursor(self): return self.c


def test_rules_need_the_root_with_the_right_part_of_speech():
    words = [(1, "visageless"), (2, "palterest"), (3, "crabbedest"), (4, "irrased"), (5, "affiancement"),
             (6, "bethoughtest")]
    wikt = [("visage", "NOUN", "The face"), ("palter", "VERB", "To talk insincerely"),
            ("crabbed", "ADJECTIVE", "Bad-tempered"), ("rased", "NOUN", "fault"),
            ("affiance", "VERB", "To betroth"), ("bethought", "ADJECTIVE", "Intended")]
    got = {p.lemma: p.definition for p in dd.propose(_Conn(words, wikt))}
    assert got["visageless"] == "Having no visage (visage: The face)"
    assert got["palterest"].startswith("(archaic) second-person singular present of palter")
    assert got["crabbedest"] == "Superlative form of crabbed — Bad-tempered"
    assert got["affiancement"].startswith("The act or result of affiancing")
    assert "irrased" not in got                  # no rule for ir- + a noun
    assert "bethoughtest" not in got             # "thou bethoughtest", not a superlative
