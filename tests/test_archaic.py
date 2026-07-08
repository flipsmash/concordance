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


# --- recency-decline (option 1: recency -> archaic) -----------------------

def test_recency_decline_flags_faded_common_word():
    flag, ev, conf = archaic.classify("Truly.", ngram_peak=1.6e-5, recency_ratio=0.01)
    assert flag == "archaic" and "faded" in ev


def test_low_peak_decline_does_not_flag():
    # cangue: decline but tiny peak -> stays current
    assert archaic.classify("A wooden collar.", ngram_peak=6e-7, recency_ratio=0.01)[0] == "current"


def test_high_recency_stays_current():
    assert archaic.classify("To speak softly.", ngram_peak=1.3e-5, recency_ratio=0.65)[0] == "current"


# --- confidence -----------------------------------------------------------

def test_recency_only_is_low_confidence():
    _, _, conf = archaic.classify("A servant.", ngram_peak=1e-5, recency_ratio=0.01)
    assert conf == 0.5                                 # the review queue


def test_explicit_label_is_high_confidence():
    _, _, conf = archaic.classify("Obsolete form of foo.")
    assert conf >= 0.9


def test_corroborating_signals_boost_confidence():
    # def-label archaic AND recency both at tier 2 -> nudge above the label's 0.9
    _, _, conf = archaic.classify("Archaic form of foo.", ngram_peak=1e-5, recency_ratio=0.01)
    assert conf > 0.9
