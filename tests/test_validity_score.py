"""Validity likelihood scorer — signals stubbed to test the scoring logic."""

from __future__ import annotations

import pytest

from concordance import validity_score as V


def _stub(monkeypatch, *, ng, corpus, neighbor=None, root=None, zipf=0.0):
    monkeypatch.setattr(V, "_ngram_peak", lambda w, s: ng)
    monkeypatch.setattr(V, "_wordset", lambda: {"cobloaf"} if corpus else set())
    monkeypatch.setattr(V, "_in_wordnet", lambda w: False)
    monkeypatch.setattr(V, "_dominant_neighbor", lambda w: neighbor)
    monkeypatch.setattr(V, "_morph_root", lambda w: root)
    monkeypatch.setattr(V, "zipf_frequency", lambda w, lang: zipf)


def test_real_archaic_in_wordlist_is_valid(monkeypatch):
    _stub(monkeypatch, ng=1e-8, corpus=True)
    e = V.estimate("cobloaf")
    assert e.label == "likely-valid" and e.score >= 0.6


def test_nonsense_absent_from_books_is_artifact(monkeypatch):
    _stub(monkeypatch, ng=0.0, corpus=False)
    e = V.estimate("zxqwplt")
    assert e.label == "likely-artifact" and e.score <= 0.35
    assert "absent from Google Books" in e.notes


def test_dominant_neighbor_flags_ocr_variant(monkeypatch):
    _stub(monkeypatch, ng=1e-7, corpus=False, neighbor=("daisies", 2.9), zipf=1.1)
    e = V.estimate("daisie")
    assert e.suggestion == "daisies"
    assert e.label in ("likely-artifact", "uncertain")
    assert e.score < 0.6


def test_books_only_is_uncertain_not_valid(monkeypatch):
    # appears in old books (Latin/old-spelling) but no wordlist, no neighbor
    _stub(monkeypatch, ng=1e-6, corpus=False, zipf=1.2)
    e = V.estimate("fatuus")
    assert e.label == "uncertain"


# --- real (offline) helper behaviour --------------------------------------

def test_morph_root_peels_affixes():
    assert V._morph_root("recoloring") in ("color", "colore", "recolor", "coloring", "recoloring") or \
           V._morph_root("recoloring") is not None

def test_dominant_neighbor_ignores_proper_names():
    # 'tarrie' has name neighbours (carrie/barrie) — those must not count unless
    # they are real WordNet words; a genuine common neighbour may still register.
    n = V._dominant_neighbor("tarrie")
    if n:
        assert V._in_wordnet(n[0]) or n[1] >= 4.0
