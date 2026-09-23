"""Archaic-currency flag + confidence (§ difficulty).

An ordinal — current < dated < archaic < obsolete — with a 0-1 confidence, because
the signals differ sharply in reliability:

  * a register label in the definition ("Obsolete form of ...", "(archaic)") or
    the related project's vocab.wiktionary is_archaic/is_obsolete — HIGH confidence.

No Google Books signal. A recency-decline test (peak >= 1e-6 and recent/peak <
0.15 -> archaic @0.5) used to be the third input and was removed 2026-09-23 as
measured-uninformative: among words Wiktionary covers, those it flagged were LESS
often Wiktionary-tagged archaic/obsolete than those it passed, in every peak era
(<1700 16.6% vs 18.3%; 1700s 9.8% vs 16.8%; 1800s 5.8% vs 11.0%). Peak year can't
rescue it -- the pre-1800 Ngram corpus is small and OCR/Latin-noisy enough that
modern words "peak" there (aileron, icosahedron, hypersurface). It was the sole
evidence for ~4.5k "archaic" labels. A windowed 1800-1900 vs 2000-2019 decline
from real timeseries (see the planned bulk Ngram download) could bring a print
signal back; words with no dictionary label are the queue for a future LLM pass.
"""

from __future__ import annotations

import re

_TIERS = ("current", "dated", "archaic", "obsolete")
_OBSOLETE_RE = re.compile(r"\bobsolete\b", re.IGNORECASE)
_ARCHAIC_RE = re.compile(r"\barchaic\b", re.IGNORECASE)
_DATED_RE = re.compile(r"\b(dated|old-fashioned)\b", re.IGNORECASE)


def _def_tier(definition: str) -> int:
    d = definition or ""
    if _OBSOLETE_RE.search(d):
        return 3
    if _ARCHAIC_RE.search(d):
        return 2
    if _DATED_RE.search(d):
        return 1
    return 0


def classify(definition: str, wik_archaic: bool = False,
             wik_obsolete: bool = False) -> tuple[str, str, float]:
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

    if not signals:
        return "current", "", 0.9

    tier = max(t for t, _, _ in signals)
    at_tier = [(c, lbl) for t, c, lbl in signals if t == tier]
    conf = max(c for c, _ in at_tier)
    if len(at_tier) > 1:
        conf = min(0.98, conf + 0.05)            # corroborating signals agree
    evidence = "; ".join(lbl for _, _, lbl in signals)
    return _TIERS[tier], evidence, round(conf, 2)
