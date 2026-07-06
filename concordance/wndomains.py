"""WordNet Domains → USAS prior (§ taxonomy).

Uses Brian's licensed WordNet-Domains data (git-ignored under
wordnet-domains-sentiwords/) as a high-precision domain prior for the USAS
classifier. WND is keyed to Princeton WordNet 1.6 synset offsets, so we bridge to
lemmas once (via the WN1.6 data files) and cache a `lemma -> {domains}` lexicon;
after that no WordNet version juggling is needed — we match our words by lemma.

`build_lexicon()` regenerates the cache from the licensed data; `load()` reads it;
`usas_prior(lemma)` maps a word's domains onto USAS category codes via the
crosswalk below. The prior is advisory — it seeds/validates the LLM classifier,
it is not the sole authority.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

LEXICON_PATH = Path("wordnet-domains-sentiwords/lemma_domains.tsv")  # git-ignored (derived from licensed data)

# WordNet-Domains label -> USAS code. Curated; None = no clean USAS home (skip).
_WND_TO_USAS = {
    # body & medicine
    "anatomy": "B1", "physiology": "B1", "health": "B2", "psychiatry": "B2",
    "medicine": "B3", "surgery": "B3", "dentistry": "B3", "pharmacy": "B3",
    "radiology": "B3", "veterinary": "B3", "body_care": "B4",
    # clothing / adornment
    "fashion": "B5", "jewellery": "B5",
    # arts & crafts
    "art": "C1", "drawing": "C1", "painting": "C1", "sculpture": "C1",
    "graphic_arts": "C1", "plastic_arts": "C1", "photography": "C1",
    "artisanship": "C1", "dance": "C1", "heraldry": "C1",
    # food & farming
    "food": "F1", "gastronomy": "F1", "agriculture": "F4",
    "animal_husbandry": "F4", "fishing": "F4",
    # government / law / war
    "administration": "G1.1", "diplomacy": "G1.1", "politics": "G1.2",
    "law": "G2.1", "military": "G3",
    # architecture & home
    "architecture": "H1", "buildings": "H1", "town_planning": "H1",
    "home": "H4", "furniture": "H5",
    # money & commerce
    "money": "I1", "finance": "I1", "banking": "I1", "economy": "I1",
    "exchange": "I1", "insurance": "I1", "tax": "I1", "numismatics": "I1",
    "commerce": "I2", "enterprise": "I2", "book_keeping": "I2", "industry": "I4",
    # entertainment / sport / games
    "free_time": "K1", "tourism": "K1", "philately": "K1",
    "music": "K2", "acoustics": "K2", "theatre": "K4", "cinema": "K4",
    "sport": "K5.1", "archery": "K5.1", "athletics": "K5.1", "badminton": "K5.1",
    "baseball": "K5.1", "basketball": "K5.1", "bowling": "K5.1", "boxing": "K5.1",
    "cricket": "K5.1", "cycling": "K5.1", "diving": "K5.1", "fencing": "K5.1",
    "football": "K5.1", "golf": "K5.1", "hockey": "K5.1", "mountaineering": "K5.1",
    "racing": "K5.1", "rowing": "K5.1", "rugby": "K5.1", "skating": "K5.1",
    "skiing": "K5.1", "soccer": "K5.1", "swimming": "K5.1", "table_tennis": "K5.1",
    "tennis": "K5.1", "volleyball": "K5.1", "wrestling": "K5.1", "hunting": "K5.1",
    "card": "K5.2", "chess": "K5.2", "betting": "K5.2", "play": "K5.2",
    # life & living things
    "biology": "L1", "genetics": "L1", "paleontology": "L1", "biochemistry": "L1",
    "animals": "L2", "zoology": "L2", "entomology": "L2", "plants": "L3",
    # movement / transport
    "transport": "M3", "vehicles": "M3", "railway": "M3", "nautical": "M4",
    "aviation": "M5", "astronautics": "M5",
    # numbers & measurement
    "number": "N1", "mathematics": "N2", "geometry": "N2", "statistics": "N2",
    "metrology": "N3",
    # substances / objects / physical attributes
    "electricity": "O3", "electronics": "O3", "electrotechnology": "O3",
    "color": "O4.3", "gas": "O1.3",
    # education
    "pedagogy": "P1", "school": "P1", "university": "P1",
    # language & communication
    "grammar": "Q3", "linguistics": "Q3", "philology": "Q3", "literature": "Q4.1",
    "publishing": "Q1.2", "post": "Q1.2", "telecommunication": "Q1.3",
    "telegraphy": "Q1.3", "telephony": "Q1.3", "tv": "Q4.3", "radio": "Q4.3",
    # social / people / religion
    "sociology": "S1.1", "social": "S1.1", "social_science": "S1.1",
    "anthropology": "S1.1", "ethnology": "S1.1", "person": "S2", "sexuality": "S3.2",
    "religion": "S9", "theology": "S9", "roman_catholic": "S9", "mythology": "S9",
    "occultism": "S9", "paranormal": "S9", "astrology": "S9", "folklore": "S9",
    # time / world / psych / science
    "time_period": "T1.3", "astronomy": "W1", "geography": "W3", "geology": "W3",
    "earth": "W3", "oceanography": "W3", "topography": "W3", "archaeology": "W3",
    "meteorology": "W4", "environment": "W5",
    "psychology": "X1", "psychoanalysis": "X1", "psychological_features": "X1",
    "philosophy": "X2.1", "quality": "A5",
    "physics": "Y1", "chemistry": "Y1", "atomic_physic": "Y1", "optics": "Y1",
    "engineering": "Y1", "mechanics": "Y1", "hydraulics": "Y1",
    "applied_science": "Y1", "pure_science": "Y1", "computer_science": "Y2",
    # deliberately unmapped (no clean USAS home): humanities, history, sub
}

_POS_FILE = {"noun": "n", "verb": "v", "adj": "a", "adv": "r"}
_lexicon: dict[str, set[str]] | None = None


def build_lexicon(wn16_dict_dir: str | Path, wnd_file: str | Path,
                  out_path: str | Path = LEXICON_PATH) -> int:
    """Bridge WordNet-1.6 lemmas to their WND domains and cache lemma->domains.
    `wnd_file` must be the WN1.6-aligned domains file (wn-domains-2.0-*)."""
    wn16_dict_dir = Path(wn16_dict_dir)
    off_lemmas: dict[tuple[str, str], list[str]] = {}
    for fn, pos in _POS_FILE.items():
        for line in open(wn16_dict_dir / f"data.{fn}", encoding="latin-1"):
            if not line[:8].isdigit():
                continue
            pre = line.split(" | ")[0].split()
            wcnt = int(pre[3], 16)
            lemmas = [re.sub(r"\(.*?\)", "", pre[4 + 2 * i].lower()).replace("_", " ").strip()
                      for i in range(wcnt)]
            off_lemmas[(pre[0], pos)] = lemmas

    lemma_dom: dict[str, set[str]] = {}
    for line in open(wnd_file):
        key, _, val = line.strip().partition("\t")
        off, _, pos = key.partition("-")
        doms = [d for d in val.split() if d and d != "factotum"]
        if not doms:
            continue
        for lem in off_lemmas.get((off, pos), []):
            lemma_dom.setdefault(lem, set()).update(doms)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for lem in sorted(lemma_dom):
            f.write(f"{lem}\t{' '.join(sorted(lemma_dom[lem]))}\n")
    return len(lemma_dom)


def load(path: str | Path = LEXICON_PATH) -> dict[str, set[str]]:
    global _lexicon
    if _lexicon is None:
        p = Path(path)
        _lexicon = {}
        if p.exists():
            for line in p.open(encoding="utf-8"):
                lem, _, doms = line.rstrip("\n").partition("\t")
                _lexicon[lem] = set(doms.split())
    return _lexicon


def domains_for(lemma: str) -> set[str]:
    return load().get(lemma.strip().lower(), set())


def usas_prior(lemma: str) -> Counter:
    """USAS codes suggested by the word's WND domains, weighted by how many domains
    point at each (a light confidence signal)."""
    codes: Counter = Counter()
    for d in domains_for(lemma):
        code = _WND_TO_USAS.get(d)
        if code:
            codes[code] += 1
    return codes
