"""Archaic-currency ordinal + confidence (pure; DB path exercised live)."""

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
    assert archaic.classify("Obsolete spelling of foo.", wik_archaic=True)[0] == "obsolete"


def test_rare_but_current_word_stays_current():
    flag, evidence, conf = archaic.classify("A large warhorse of a medieval knight.")
    assert flag == "current" and evidence == ""


def test_word_boundary_avoids_false_positives():
    assert archaic.classify("An undated manuscript.")[0] == "current"


# --- no print-trend signal ---------------------------------------------

def test_print_decline_alone_no_longer_flags():
    # the removed Ngram recency test was measured uninformative (see archaic.py);
    # classify() no longer even accepts print data
    import inspect
    assert "recency_ratio" not in inspect.signature(archaic.classify).parameters
    assert archaic.classify("Truly.")[0] == "current"


# --- confidence -----------------------------------------------------------


def test_explicit_label_is_high_confidence():
    _, _, conf = archaic.classify("Obsolete form of foo.")
    assert conf >= 0.9


def test_corroborating_signals_boost_confidence():
    # def-label archaic AND wiktionary archaic both at tier 2 -> nudge above the label's 0.9
    _, _, conf = archaic.classify("Archaic form of foo.", wik_archaic=True)
    assert conf > 0.9
