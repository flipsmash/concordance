"""Archaic-currency flag (§ difficulty).

An ordinal — current < dated < archaic < obsolete — estimating whether the word
is still in living use. A word being *rare* is not the same as being *archaic*
(cangue is a current, rare word), so this uses only currency signals:

  * register labels in the definition we already fetched ("Obsolete form of ...",
    "(archaic)", "dated") — these catch the hyper-rare Shakespearean tail
    (abhominable, villany, enow) that dictionaries-as-data miss;
  * the related project's vocab.wiktionary is_archaic / is_obsolete booleans —
    high precision on common-to-mid vocabulary.

(Google-Books recency-decline is a natural third signal but needs a network call
per word; left as a future enhancement.)
"""

from __future__ import annotations

import re

_TIERS = ("current", "dated", "archaic", "obsolete")
_OBSOLETE_RE = re.compile(r"\bobsolete\b", re.IGNORECASE)
_ARCHAIC_RE = re.compile(r"\barchaic\b", re.IGNORECASE)
_DATED_RE = re.compile(r"\b(dated|old-fashioned)\b", re.IGNORECASE)


def _def_tier(definition: str) -> int:
    """0 current .. 3 obsolete, from a register label in the gloss."""
    d = definition or ""
    if _OBSOLETE_RE.search(d):
        return 3
    if _ARCHAIC_RE.search(d):
        return 2
    if _DATED_RE.search(d):
        return 1
    return 0


def classify(definition: str, wik_archaic: bool = False, wik_obsolete: bool = False) -> tuple[str, str]:
    """Return (flag, evidence). Strongest signal wins."""
    tier, evidence = 0, []
    dt = _def_tier(definition)
    if dt:
        tier = dt
        evidence.append(f"definition label: {_TIERS[dt]}")
    if wik_obsolete and tier < 3:
        tier = 3
        evidence.append("wiktionary: obsolete")
    elif wik_archaic and tier < 2:
        tier = 2
        evidence.append("wiktionary: archaic")
    return _TIERS[tier], "; ".join(evidence)
