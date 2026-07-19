"""Lightweight priority re-exposure for quiz spaced repetition (§ quizzing).

Not full SM-2 and not the mastery-tracking system (both explicitly deferred)
-- just enough state to bias which words a quiz pulls from next: a missed
word should resurface sooner than one just answered correctly. Streak-based
exponential backoff on a correct answer, a short fixed cooldown on a miss,
both scaled by a frequency knob (loose/normal/tight).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

Frequency = Literal["loose", "normal", "tight"]

_BASE_INTERVAL = timedelta(days=1)
_MAX_INTERVAL = timedelta(days=30)
_SHORT_INTERVAL = timedelta(hours=8)

# tight -> shorter intervals -> words resurface sooner/more often;
# loose -> longer intervals -> words resurface later/less often.
_FREQUENCY_MULTIPLIERS: dict[Frequency, float] = {"loose": 2.0, "normal": 1.0, "tight": 0.5}


@dataclass
class ReviewUpdate:
    streak: int
    next_eligible_at: datetime


def next_review(streak: int, is_correct: bool, frequency: Frequency = "normal",
                 now: datetime | None = None) -> ReviewUpdate:
    """The new streak + next_eligible_at after answering a word correctly or
    incorrectly. `streak` is the value BEFORE this answer (0 for a word never
    seen, or whose last answer was a miss)."""
    now = now or datetime.now(timezone.utc)
    multiplier = _FREQUENCY_MULTIPLIERS[frequency]
    if is_correct:
        new_streak = streak + 1
        interval = min(_BASE_INTERVAL * (2 ** new_streak), _MAX_INTERVAL) * multiplier
    else:
        new_streak = 0
        interval = _SHORT_INTERVAL * multiplier
    return ReviewUpdate(streak=new_streak, next_eligible_at=now + interval)
