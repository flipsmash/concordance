"""ARPAbet -> IPA conversion (§ audio pronunciation).

ARPAbet is the CMU Pronouncing Dictionary's phoneme set (and what Wordnik's
`rawType == "arpabet"` pronunciations use — same underlying data). It's a small,
closed, well-documented inventory for American English, so this is a deterministic
lookup table, not a guess. Vowel phones carry a trailing stress digit
(0 = none, 1 = primary, 2 = secondary) which becomes an IPA stress mark placed
before the syllable rather than attached to the vowel.
"""

from __future__ import annotations

import re

# Consonants: no stress marking, 1:1 IPA equivalents.
_CONSONANTS = {
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p",
    "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}

# Vowels (stress digit stripped before lookup): standard General American IPA.
_VOWELS = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ",
    "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
}

_PHONE_RE = re.compile(r"[A-Z]+[0-2]?")


def to_ipa(arpabet: str) -> str | None:
    """'R EY1 K ER0' -> 'ɹˈeɪkɝ'. Returns None on any unrecognized phone (fail
    closed rather than emit a partial/wrong transcription).

    The stress mark is placed before the syllable's full onset (the run of
    consonants immediately preceding the stressed vowel), not immediately before
    the vowel itself — ARPAbet marks stress per-vowel with no syllable-boundary
    info, but IPA convention (and Azure's documented fallback rule for stress
    without explicit syllable dots) reads the mark as starting the syllable,
    onset consonants included. Getting this wrong risks the consonant cluster
    being attributed to the wrong syllable by the synthesizer.
    """
    phones = arpabet.strip().split()
    if not phones:
        return None
    # (ipa_symbol, is_vowel, stress_mark_or_None)
    symbols: list[tuple[str, bool, str | None]] = []
    for ph in phones:
        m = re.match(r"^([A-Z]+)([0-2])?$", ph)
        if not m:
            return None
        base, stress = m.group(1), m.group(2)
        if base in _VOWELS:
            mark = "ˈ" if stress == "1" else "ˌ" if stress == "2" else None
            symbols.append((_VOWELS[base], True, mark))
        elif base in _CONSONANTS:
            symbols.append((_CONSONANTS[base], False, None))
        else:
            return None

    out: list[str] = []
    onset_start = 0  # index into `out` marking the start of the current consonant run
    for sym, is_vowel, mark in symbols:
        if mark:
            out.insert(onset_start, mark)
        out.append(sym)
        if is_vowel:
            onset_start = len(out)  # next consonant run (if any) starts fresh, after this vowel
    return "".join(out)
