"""Ex-ante difficulty scalar (§ difficulty).

A 0-100 estimate of how hard a word is for a well-read adult, with the *reason*
kept first-class (stored factor contributions — "hard because..."). No quiz data
yet, so this is a principled ex-ante blend, not a fitted model; IRT calibration
comes later when quiz responses exist.

Signals (all already in the DB):
  rarity   dominant — wordfreq Zipf, scaled against the collection's own floor,
           plus a bump for words absent from Google Books entirely;
  archaic  obsolete/archaic/dated, WEIGHTED BY the archaic confidence (so noisy
           recency-only flags nudge less than explicit register labels);
  domain   a specialised concrete-domain term (nautical, medicine...) is harder;
  morph    morphological transparency EASES difficulty (un+geniture+d is inferable).
"""

from __future__ import annotations

import math

# Concrete/subject USAS top fields (specialised); the rest (A E N Q S T X, Z) are
# general/expressive and don't earn a domain-specificity bump.
DOMAIN_FIELDS = set("BCFGHIKLMOWY")

# Zipf ceiling for the rarity scale: the pipeline floor is ~3.5, so a hair above
# it maps the least-rare surviving words near 0 and the corpus spans 0..1.
_ZIPF_CEIL = 4.0

_ARCHAIC_BASE = {"obsolete": 0.25, "archaic": 0.18, "dated": 0.06, "current": 0.0}


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _rarity(zipf: float, ngram_peak: float | None) -> float:
    """Blend web rarity (Zipf, saturates below its coverage floor) with print
    rarity (Ngram, log-scaled) so ultra-rare words don't all pile up at the top."""
    r_web = _clamp((_ZIPF_CEIL - zipf) / _ZIPF_CEIL)
    if ngram_peak and ngram_peak > 0:
        r_print = _clamp((-math.log10(ngram_peak) - 5.0) / 6.0)   # 1e-5 -> 0 .. 1e-11 -> 1
    else:
        r_print = 1.0                        # absent from Google Books entirely
    return 0.65 * r_web + 0.35 * r_print


def score(zipf: float, ngram_peak: float | None, archaic: str = "current",
          archaic_conf: float | None = None, has_domain: bool = False,
          morph_transparent: bool = False) -> tuple[int, dict]:
    """Return (difficulty 0-100, factors dict incl. a human 'why')."""
    factors: dict = {"zipf": round(zipf, 2)}

    rarity = _rarity(zipf, ngram_peak)       # zipf (web) blended with Ngram (print)
    factors["rarity"] = round(rarity, 3)

    arch = _ARCHAIC_BASE.get(archaic, 0.0) * (archaic_conf if archaic_conf is not None else 1.0)
    factors["archaic"] = round(arch, 3)

    domain = 0.06 if has_domain else 0.0
    factors["domain"] = domain

    morph = -0.10 if morph_transparent else 0.0
    factors["morph"] = morph

    total = _clamp(rarity + arch + domain + morph)
    factors["why"] = _why(zipf, ngram_peak, archaic, arch, domain, morph)
    return round(total * 100), factors


def _why(zipf, ngram_peak, archaic, arch, domain, morph) -> str:
    hard, easy = [], []
    if zipf <= 1.0:
        hard.append(f"very rare (zipf {zipf:.1f})")
    elif zipf <= 2.5:
        hard.append(f"rare (zipf {zipf:.1f})")
    if ngram_peak == 0:
        hard.append("absent from print (Google Books)")
    if arch > 0:
        hard.append(f"{archaic}")
    if domain > 0:
        hard.append("specialised domain")
    if morph < 0:
        easy.append("morphologically transparent")
    s = "hard: " + (", ".join(hard) if hard else "—")
    if easy:
        s += "; eased by: " + ", ".join(easy)
    return s
