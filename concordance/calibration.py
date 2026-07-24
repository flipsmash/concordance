"""Personalized quiz-response difficulty calibration (§ calibration).

A per-(user, word) adjustment to the ex-ante difficulty score
(`concordance/difficulty.py`), from that user's own FIRST exposure to the
word in a quiz. Deliberately NOT a population-level IRT calibration: with
one dominant rater (see `compute_personal_difficulty`'s own docstring in
db.py), response volume only ever tells you that person's own relative
gaps, never identifies "true" item difficulty the way a real multi-rater
IRT fit would. This is closer to "ex-ante score plus one bit of first-hand
evidence per word" than to a rigorous psychometric model, and is scoped
and named that way throughout rather than oversold.

The model: a fixed-guessing-floor Rasch form with a FIXED reference
ability (theta=0, never estimated) --

    P(correct | b_word, c_q) = c_q + (1 - c_q) * sigmoid(-b_word)

-- and a single step of gradient ascent on the log-likelihood per first
exposure (the same derivation Elo's own update rule comes from):

    b_new = b0 - eta * (y - P0)

Since each (user, word) pair contributes at most one first-exposure
response ever (see db.py), there is no accumulating sample size to shrink
over per item -- every personalized value is always exactly "ex-ante,
nudged once" or "ex-ante, untouched." eta/scale are accordingly the only
two tunable constants, and are meant to be hand-tuned (via app_settings),
not fit -- there isn't enough independent data to fit them either, for the
same one-rater reason.
"""

from __future__ import annotations

import math

DEFAULT_SCALE = 16.0   # logit units per ex-ante difficulty point: (100-50)/SCALE ~= 3, a typical Rasch b-parameter range
DEFAULT_ETA = 1.0      # max logit-unit shift from a single maximally-surprising response


def _sigmoid(x: float) -> float:
    # Guard against overflow on an extreme b0 (shouldn't happen at DEFAULT_SCALE
    # for a 0-100 input, but a hand-tuned SCALE could push it there).
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def difficulty_to_logit(difficulty: float, scale: float = DEFAULT_SCALE) -> float:
    """0-100 ex-ante difficulty -> logit-scale b_word."""
    return (difficulty - 50.0) / scale


def logit_to_difficulty(b: float, scale: float = DEFAULT_SCALE) -> float:
    """Inverse of difficulty_to_logit, clamped back into the 0-100 range the
    rest of the app already expects difficulty values to live in."""
    return max(0.0, min(100.0, b * scale + 50.0))


def guessing_floor(question_type: str, option_count: int) -> float:
    """The structural chance-correct probability for a question, from the
    ACTUAL assembled option/set count -- not nominal request config, which
    can differ: _build_mc_payload can reduce option count when
    select_mc_distractors returns fewer candidates than requested, and
    _build_matching_payload accepts sets below the nominal matching_set_size.
    For matching, 1/option_count is the exact marginal per-pair probability
    under a uniformly random guess, not an approximation."""
    if question_type == "true_false":
        return 0.5
    if option_count <= 0:
        return 0.0
    return 1.0 / option_count


def response_probability(b_word: float, c_q: float) -> float:
    """P(correct | b_word, c_q) under the fixed-guessing-floor, theta=0 model."""
    return c_q + (1.0 - c_q) * _sigmoid(-b_word)


def update_rating(b0: float, is_correct: bool, c_q: float, eta: float = DEFAULT_ETA) -> float:
    """One gradient-ascent step on the log-likelihood, applied to a single
    first-exposure response -- see module docstring for the derivation and
    why there's no accumulation/shrinkage math needed here."""
    p0 = response_probability(b0, c_q)
    y = 1.0 if is_correct else 0.0
    return b0 - eta * (y - p0)
