"""Spaced-repetition interval computation. Pure -- no database."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from concordance import spaced_repetition as sr

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_correct_answer_grows_streak_and_interval():
    r0 = sr.next_review(streak=0, is_correct=True, now=_NOW)
    assert r0.streak == 1
    assert r0.next_eligible_at == _NOW + timedelta(days=2)  # 1 day * 2^1

    r1 = sr.next_review(streak=1, is_correct=True, now=_NOW)
    assert r1.streak == 2
    assert r1.next_eligible_at == _NOW + timedelta(days=4)  # 1 day * 2^2


def test_correct_answer_interval_caps_at_max():
    r = sr.next_review(streak=10, is_correct=True, now=_NOW)  # 2^11 days, way over the cap
    assert r.next_eligible_at == _NOW + timedelta(days=30)


def test_incorrect_answer_resets_streak_and_uses_short_interval():
    r = sr.next_review(streak=5, is_correct=False, now=_NOW)
    assert r.streak == 0
    assert r.next_eligible_at == _NOW + timedelta(hours=8)


def test_tight_frequency_shortens_both_intervals():
    correct = sr.next_review(streak=0, is_correct=True, frequency="tight", now=_NOW)
    assert correct.next_eligible_at == _NOW + timedelta(days=1)  # 2 days * 0.5

    incorrect = sr.next_review(streak=3, is_correct=False, frequency="tight", now=_NOW)
    assert incorrect.next_eligible_at == _NOW + timedelta(hours=4)  # 8h * 0.5


def test_loose_frequency_lengthens_both_intervals():
    correct = sr.next_review(streak=0, is_correct=True, frequency="loose", now=_NOW)
    assert correct.next_eligible_at == _NOW + timedelta(days=4)  # 2 days * 2.0

    incorrect = sr.next_review(streak=3, is_correct=False, frequency="loose", now=_NOW)
    assert incorrect.next_eligible_at == _NOW + timedelta(hours=16)  # 8h * 2.0


def test_missed_word_resurfaces_sooner_than_correctly_answered_word():
    # The whole point of "priority re-exposure": at the same streak depth, a
    # miss must always come back sooner than a hit, for every frequency.
    for freq in ("loose", "normal", "tight"):
        correct = sr.next_review(streak=2, is_correct=True, frequency=freq, now=_NOW)
        incorrect = sr.next_review(streak=2, is_correct=False, frequency=freq, now=_NOW)
        assert incorrect.next_eligible_at < correct.next_eligible_at
