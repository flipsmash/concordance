"""Archaic-currency ordinal (pure logic; DB path exercised live)."""

from __future__ import annotations

from concordance import archaic


def test_definition_labels_set_the_tier():
    assert archaic.classify("Obsolete form of abominable.")[0] == "obsolete"
    assert archaic.classify("Archaic form of enough.")[0] == "archaic"
    assert archaic.classify("(dated) A partygoer.")[0] == "dated"
    assert archaic.classify("A heavy wooden collar.")[0] == "current"


def test_wiktionary_flags_contribute():
    assert archaic.classify("A benevolent spirit.", wik_obsolete=True)[0] == "obsolete"
    assert archaic.classify("Truly; indeed.", wik_archaic=True)[0] == "archaic"


def test_strongest_signal_wins():
    # obsolete definition label outranks a mere archaic wiktionary flag
    flag, _ = archaic.classify("Obsolete spelling of foo.", wik_archaic=True)
    assert flag == "obsolete"


def test_rare_but_current_word_stays_current():
    flag, evidence = archaic.classify("A large warhorse of a medieval knight.")
    assert flag == "current" and evidence == ""


def test_evidence_is_reported():
    _, evidence = archaic.classify("Obsolete form of x.", wik_obsolete=True)
    assert "definition label: obsolete" in evidence


def test_word_boundary_avoids_false_positives():
    # 'undated' must not trip the 'dated' matcher
    assert archaic.classify("An undated manuscript.")[0] == "current"
