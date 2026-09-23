"""Ex-ante difficulty scalar (§ difficulty).

A 0-100 estimate of how hard a word is for a well-read adult, with the *reason*
kept first-class (stored factor contributions — "hard because..."). No quiz data
yet, so this is a principled ex-ante blend, not a fitted model; IRT calibration
comes later when quiz responses exist.

Signals (all already in the DB):
  rarity   dominant — ONE zipf-unit frequency on a fixed scale: wordfreq's Zipf
           where wordfreq lists the word, else its 2000-2019 Google Books
           frequency converted to zipf units (see `unified_zipf`);
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

# Fixed rarity scale in zipf units (log10 occurrences per billion words) --
# absolute constants, deliberately NOT derived from the current corpus's stats.
# CEIL: a hair above the pipeline's ~3.5 frequency floor, so the least-rare
# surviving words map near 0. FLOOR: roughly Google Books' resolution limit in
# 2000-2019 (1e-11 relative frequency ~ a handful of occurrences across those
# twenty years); below that, frequency can't meaningfully separate words.
_ZIPF_CEIL = 4.0
_ZIPF_FLOOR = -2.0

_ARCHAIC_BASE = {"obsolete": 0.25, "archaic": 0.18, "dated": 0.06, "current": 0.0}


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def print_zipf(ngram_recent: float | None) -> float | None:
    """2000-2019 Google Books relative frequency -> zipf units (clamped to the
    scale floor); None when there is no Ngram data for the word at all."""
    if ngram_recent is None:
        return None
    if ngram_recent <= 0:
        return _ZIPF_FLOOR
    return max(_ZIPF_FLOOR, math.log10(ngram_recent * 1e9))


def unified_zipf(web_zipf: float, ngram_recent: float | None,
                 root_zipf: float | None = None) -> tuple[float, str]:
    """The single zipf-unit frequency rarity is scaled from, and its source.

    wordfreq's Zipf bottoms out around 0.84 -- a 0 there means "not listed",
    not "never used", and ~2/3 of the collection sits at that 0. Below that
    cutoff, fall back to the RECENT (2000-2019) Ngram frequency in zipf units,
    which tracks wordfreq closely where both exist. Not the all-time peak: the
    pre-1800 Ngram corpus is so small a single use reads as a spike (e.g.
    "cognoscibility" peaks in 1674), and faded words are the archaic factor's
    job anyway. A transparent derivation takes its common root's Zipf when
    higher (unbuttoned -> button), as effective_zipf does."""
    if web_zipf > 0:
        z, src = web_zipf, "wordfreq"
    else:
        pz = print_zipf(ngram_recent)
        z, src = (pz, "ngram") if pz is not None else (_ZIPF_FLOOR, "unseen")
    if root_zipf is not None and root_zipf > z:
        z, src = root_zipf, "root"
    return z, src


def _rarity(zipf: float) -> float:
    return _clamp((_ZIPF_CEIL - zipf) / (_ZIPF_CEIL - _ZIPF_FLOOR))


def score(web_zipf: float, ngram_recent: float | None, ngram_peak: float | None = None,
          archaic: str = "current", archaic_conf: float | None = None,
          has_domain: bool = False, morph_transparent: bool = False,
          root_zipf: float | None = None) -> tuple[int, dict]:
    """Return (difficulty 0-100, factors dict incl. a human 'why').

    `zipf` in the factors is the unified value rarity is computed from (so the
    two always move together); `zipf_web`/`zipf_print` are the raw inputs and
    `zipf_source` says which one won (wordfreq | ngram | root | unseen)."""
    zipf, src = unified_zipf(web_zipf, ngram_recent, root_zipf)
    pz = print_zipf(ngram_recent)
    factors: dict = {"zipf": round(zipf, 2), "zipf_source": src,
                     "zipf_web": round(web_zipf, 2),
                     "zipf_print": round(pz, 2) if pz is not None else None}

    rarity = _rarity(zipf)
    factors["rarity"] = round(rarity, 3)

    arch = _ARCHAIC_BASE.get(archaic, 0.0) * (archaic_conf if archaic_conf is not None else 1.0)
    factors["archaic"] = round(arch, 3)

    domain = 0.06 if has_domain else 0.0
    factors["domain"] = domain

    morph = -0.10 if morph_transparent else 0.0
    factors["morph"] = morph

    total = _clamp(rarity + arch + domain + morph)
    factors["why"] = _why(zipf, src, ngram_peak, archaic, arch, domain, morph)
    return round(total * 100), factors


def _why(zipf, src, ngram_peak, archaic, arch, domain, morph) -> str:
    hard, easy = [], []
    via = {"ngram": ", from print", "root": ", via root"}.get(src, "")
    if zipf <= 1.0:
        hard.append(f"very rare (zipf {zipf:.1f}{via})")
    elif zipf <= 2.5:
        hard.append(f"rare (zipf {zipf:.1f}{via})")
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
