"""American Heritage Dictionary respelling -> IPA (§ audio pronunciation).

AHD's respelling system (what Wordnik's `rawType == "ahd-5"` pronunciations use)
is a documented, closed symbol set — deterministic, not a guess. Two things make
it trickier than ARPAbet:

  1. Multi-character symbols (âr, ch, o͝o, ...) require longest-match-first
     tokenization, not a plain char-by-char lookup.
  2. Stress marks are POSTFIX (follow the stressed vowel: "kăv′ĭ") where IPA is
     PREFIX (precede the syllable's onset consonants: "ˈkæv..."). Verified
     against a real, known stress pattern (pusillanimity: secondary on "pu-",
     primary on "-nim-") that ' is primary and ″ is secondary.

AHD distinguishes voiceless "th" (θ, thigh) from voiced "th" (ð, thy) by italics
in print — which looked like it would be lost in plain text, but isn't: the
digital source encodes voiced "th" as small-caps ᴛʜ (U+1D1B U+029C), a distinct,
plain-text-safe pair of characters. Verified against "swarth" -> ð, confirmed by
kaikki's own independent IPA for the same word.
"""

from __future__ import annotations

import re

# Longest symbols first so tokenization matches greedily and correctly.
_VOWELS = [
    ("âr", "ɛər"), ("îr", "ɪər"), ("ôr", "ɔːr"), ("ûr", "ɜːr"),
    # Combining marks attach to the PRECEDING character (standard Unicode
    # behavior) — verified against real data that this is "oo" + mark, not
    # "o" + mark + "o" as the visual glyph might suggest.
    ("oo" + chr(0x35d), "ʊ"),  # combining double breve below: book
    ("oo" + chr(0x35e), "uː"),  # combining double macron below: boot
    ("ā", "eɪ"), ("ä", "ɑː"), ("ē", "iː"), ("ī", "aɪ"), ("ō", "oʊ"), ("ô", "ɔː"),
    ("ă", "æ"), ("ĕ", "ɛ"), ("ĭ", "ɪ"), ("ŏ", "ɒ"), ("ŭ", "ʌ"),
    ("oi", "ɔɪ"), ("ou", "aʊ"),
    ("ər", "ər"), ("ə", "ə"),
]

_CONSONANTS = [
    # Small-caps ᴛʜ (U+1D1B U+029C) is AHD's plain-text-safe way of marking the
    # voiced "th" (ð, thy) distinctly from the voiceless one (θ, thigh, plain
    # "th") — verified against "swarth" -> /swɔːrð/, confirmed voiced by kaikki's
    # own IPA. Must be checked before plain "th" (longest/most-specific match).
    ("ᴛʜ", "ð"),
    ("ch", "tʃ"), ("hw", "hw"), ("ng", "ŋ"), ("sh", "ʃ"), ("th", "θ"), ("zh", "ʒ"),
    ("b", "b"), ("d", "d"), ("f", "f"), ("g", "ɡ"), ("h", "h"), ("j", "dʒ"),
    ("k", "k"), ("l", "l"), ("m", "m"), ("n", "n"), ("p", "p"), ("r", "ɹ"),
    ("s", "s"), ("t", "t"), ("v", "v"), ("w", "w"), ("y", "j"), ("z", "z"),
]

# Longest-match-first: multi-char vowels/consonants before single letters.
_SYMBOLS = sorted(_VOWELS + _CONSONANTS, key=lambda kv: -len(kv[0]))
_VOWEL_SET = {ipa for _, ipa in _VOWELS}

_PRIMARY = "′"
_SECONDARY = "″"
_STRESS_CHARS = {_PRIMARY, _SECONDARY}


def to_ipa(respelling: str) -> str | None:
    """'kŏn-kăv′ĭ-tē' -> 'kɒnˈkævɪtiː'. Returns None if any span of the input
    can't be matched against the known symbol set (fail closed).

    Stress marks are POSTFIX in AHD (follow the syllable they stress) but
    PREFIX in IPA (precede the syllable's onset). `last_syll_start` tracks
    where the just-completed syllable's onset began, so a stress mark can be
    inserted there; `pending_start` tracks where the *next* syllable's onset
    will begin once a new vowel appears. An explicit hyphen (a real syllable
    boundary) freezes `pending_start` at that point, so a coda consonant
    before the hyphen (e.g. the "n" in kon-CAV-ity) isn't misassigned as the
    onset of the following syllable — without this, "kŏn-kăv" would wrongly
    read as one onset cluster "nk-" instead of a closed "kon" syllable.
    """
    text = respelling.strip()
    text = re.sub(r"<[^>]+>", "", text)  # strip embedded HTML notes, e.g. "<em>when unstressed</em>"
    text = re.split(r"[,;]", text)[0].strip()  # AHD sometimes lists variant pronunciations; take the first
    text = text.replace("ˈ", "")  # defensive: strip if a bold-stress marker slipped through as this char

    out: list[str] = []
    pending_start = 0   # where the next (not-yet-seen) syllable's onset begins
    last_syll_start = 0  # where the most-recently-completed syllable's onset began
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "-":
            pending_start = len(out)
            i += 1
            continue
        if ch in _STRESS_CHARS:
            mark = "ˈ" if ch == _PRIMARY else "ˌ"
            out.insert(last_syll_start, mark)
            if pending_start >= last_syll_start:
                pending_start += 1
            i += 1
            continue
        matched = False
        for sym, ipa in _SYMBOLS:
            if text.startswith(sym, i):
                if ipa in _VOWEL_SET:
                    last_syll_start = pending_start
                out.append(ipa)
                if ipa in _VOWEL_SET:
                    pending_start = len(out)
                i += len(sym)
                matched = True
                break
        if not matched:
            return None
    return "".join(out)
