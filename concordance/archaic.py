"""Archaic-currency flag + confidence (§ difficulty).

An ordinal — current < dated < archaic < obsolete — with a 0-1 confidence, because
the signals differ sharply in reliability:

  * a register label in the definition ("Obsolete form of ...", "(archaic)") or
    the related project's vocab.wiktionary is_archaic/is_obsolete — HIGH confidence;
  * Google-Books recency-decline (a word that was genuinely common and faded) —
    a real but NOISY signal: it can't be numerically separated from words that are
    merely uncommon-today-but-current (congeal, prorogue), so it earns only LOW
    confidence. Those low-confidence rows are the queue for future manual/LLM review.

Rare != archaic — a low-peak word that declined is just a rare/historical-referent
word (cangue), so recency only fires above a peak floor.
"""

from __future__ import annotations

import re

_TIERS = ("current", "dated", "archaic", "obsolete")
_OBSOLETE_RE = re.compile(r"\bobsolete\b", re.IGNORECASE)
_ARCHAIC_RE = re.compile(r"\barchaic\b", re.IGNORECASE)
_DATED_RE = re.compile(r"\b(dated|old-fashioned)\b", re.IGNORECASE)

_RECENCY_MIN_PEAK = 1e-6
_RECENCY_MAX_RATIO = 0.15


def _def_tier(definition: str) -> int:
    d = definition or ""
    if _OBSOLETE_RE.search(d):
        return 3
    if _ARCHAIC_RE.search(d):
        return 2
    if _DATED_RE.search(d):
        return 1
    return 0


def classify(definition: str, wik_archaic: bool = False, wik_obsolete: bool = False,
             ngram_peak: float | None = None,
             recency_ratio: float | None = None) -> tuple[str, str, float]:
    """Return (flag, evidence, confidence). Strongest tier wins; confidence is that
    of the strongest signal at the winning tier (corroboration nudges it up)."""
    signals: list[tuple[int, float, str]] = []   # (tier, confidence, label)

    dt = _def_tier(definition)
    if dt == 3:
        signals.append((3, 0.95, "definition label: obsolete"))
    elif dt == 2:
        signals.append((2, 0.9, "definition label: archaic"))
    elif dt == 1:
        signals.append((1, 0.85, "definition label: dated"))

    if wik_obsolete:
        signals.append((3, 0.9, "wiktionary: obsolete"))
    elif wik_archaic:
        signals.append((2, 0.85, "wiktionary: archaic"))

    if (ngram_peak is not None and recency_ratio is not None
            and ngram_peak >= _RECENCY_MIN_PEAK and recency_ratio < _RECENCY_MAX_RATIO):
        signals.append((2, 0.5, f"faded in print (recency {recency_ratio:.2f})"))

    if not signals:
        return "current", "", 0.9

    tier = max(t for t, _, _ in signals)
    at_tier = [(c, lbl) for t, c, lbl in signals if t == tier]
    conf = max(c for c, _ in at_tier)
    if len(at_tier) > 1:
        conf = min(0.98, conf + 0.05)            # corroborating signals agree
    evidence = "; ".join(lbl for _, _, lbl in signals)
    return _TIERS[tier], evidence, round(conf, 2)
