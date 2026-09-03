"""GCIDE (Webster's 1913) diacritical respelling -> IPA (§ audio pronunciation).

GCIDE's diacritical marks (what Wordnik's `rawType == "gcide-diacritical"`
pronunciations use) are documented in gcide's own `pronunc.txt`
(github.com/johncf/gcide) -- every mapping below is grounded in that file's
worked examples, cross-checked against real `word.wordnik_pron_raw` rows
where the file gave no example of its own:

  ā/ăsl (macron, incl. "semilong" unstressed long a -- same char here)  eɪ
  ă     (breve, short a)                                                æ
  ȧ     (dot-above, "short a" -- the file's own author flags "again" vs
         "mass"/"mat" sounding different under the same mark to them;
         treated the same as ă, imprecise but never wrong-language)     æ
  â     (circumflex, "only in syllables closed by r": care, share)      ɛər
  ä     (umlaut: arm, far, mar)                                         ɑː
  ē     (macron, long e)                                                iː
  ĕ     (breve, short e)                                                ɛ
  ẽr    (tilde, "the e before r" -- 100% of real rows pair it with a
         following r: fern, girtline)                                   ɜr
  ī     (macron, long i)                                                aɪ
  ĭ     (breve, short i)                                                ɪ
  ō     (macron, long o)                                                oʊ
  ŏ     (breve, short o)                                                ɒ
  ô     (circumflex, "only before r": orb, lord, deform)                ɔː
  ū     (macron, long u)                                                juː
  ŭ     (breve, short u)                                                ʌ
  ûr    (circumflex, "the u before r" -- same convention as ẽr)         ɜr
  ụ     (dot-below -- empirically only the "-ful" suffix: timeful,
         batful, dreamful)                                              ə
  ṳ     (diaeresis-below -- empirically "crewet"/"rewe", i.e.
         cruet/rue)                                                     uː
  ṉ     (submacron, "ng before a consonant" -- confirmed by
         handkercher/lingot/fenks/tunk/bunkbed)                         ŋ
  oi/ou (plain digraphs -- pronunc.txt gives "ou" [town, browse] as its
         own explicit undecorated example; "oi" [toy, boil] is the same
         convention, confirmed empirically: toysome/kreutzer/chowter)   ɔɪ/aʊ

Plain, undiacriticked a/e/i/o/u are GCIDE's "italicised"/obscured vowel with
the diacritic lost in this ASCII/Unicode export -- mapped to schwa per the
file's own note #3: "the indefinite value ... is not used, the same sound
being represented by symbols like short u, or sometimes other vowels."
Plain y is NOT folded into that bucket: GCIDE's y-breve ("sounds ... like an
unstressed long e") is what a bare word-final y renders as once its own
breve is lost in this export, so plain y -> iː, not schwa.

An apostrophe marks an elided vowel before a syllabic consonant (b'l, 'n,
'm, d'l -- boxen, legitimism, hamshackle, cruddle) -> a schwa insert.

Stress/syllables: " is primary (heavy), ` is secondary (light), * is a
plain syllable break, - is a literal hyphenated-word boundary. Postfix in
GCIDE (follows the stressed syllable) but PREFIX in IPA (precedes the
syllable's onset) -- same postfix->prefix shape as ahd.py, but GCIDE's own
key states "where an accent occurs, no other syllable break is used": the
stress mark itself doubles as the boundary to the next syllable, so (unlike
ahd.py) the next syllable's onset resets to right after the mark, not a
one-position nudge -- verified against "clocklike" (klŏk"līk`), which needs
the coda /k/ kept with the first syllable: ˈklɒkˌlaɪk, not ˈklɒˌklaɪk.

Fails closed (returns None) on:
  - '?' / '#' -- this Wordnik export's own OCR/transcription-failure
    placeholder (confirmed empirically: ~45% of all gcide-diacritical rows
    contain at least one -- by far the largest source of misses, not
    something any symbol table can recover).
  - A bare capital N (GCIDE's French nasal vowel marker, e.g. "rente") --
    not English, not this cascade's problem.
  - A leading hyphen (e.g. cabalist -> "-lĭst") -- GCIDE elides the shared
    prefix on a derived word and shows only the changed suffix, reusing the
    base word's own pronunciation for the rest. NOT a complete
    pronunciation: synthesizing it against the full headword text would
    anchor Azure's <phoneme> tag to the wrong, truncated sound, and there's
    no safe way to reconstruct the elided prefix from this string alone.
  - A handful of entries that give the plain English spelling as a
    stand-in (e.g. "consound" -> "sound") instead of an actual respelling.

A "; F. ..." / "; It. ..." trailing clause (a foreign-language alternate)
and a " or " listing several alternates both just take the first clause,
same "take the first" convention ahd.py uses for AHD's own variant lists.

Empirically recovers 387 of the 918 words that had `wordnik_pron_raw` under
this rawType but no `word.ipa` as of 2026-09-03 (the rest are the '?'/'#'/
elided-prefix cases above -- genuinely unrecoverable from this data, not a
gap this module's symbol table could close).
"""

from __future__ import annotations

import re

_VOWELS = [
    ("ẽr", "ɜr"), ("ûr", "ɜr"),
    ("ā", "eɪ"), ("ă", "æ"), ("ȧ", "æ"), ("â", "ɛər"), ("ä", "ɑː"),
    ("ē", "iː"), ("ĕ", "ɛ"),
    ("ī", "aɪ"), ("ĭ", "ɪ"),
    ("ō", "oʊ"), ("ŏ", "ɒ"), ("ô", "ɔː"),
    ("ū", "juː"), ("ŭ", "ʌ"),
    ("ụ", "ə"), ("ṳ", "uː"),
    ("oi", "ɔɪ"), ("ou", "aʊ"),
]

_CONSONANTS = [
    ("ṉ", "ŋ"),
    ("ch", "tʃ"), ("sh", "ʃ"), ("zh", "ʒ"), ("th", "θ"),
    ("b", "b"), ("d", "d"), ("f", "f"), ("g", "ɡ"), ("h", "h"), ("j", "dʒ"),
    ("k", "k"), ("l", "l"), ("m", "m"), ("n", "n"), ("p", "p"), ("r", "ɹ"),
    ("s", "s"), ("t", "t"), ("v", "v"), ("w", "w"), ("y", "j"), ("z", "z"),
]

_PLAIN_VOWEL_SCHWA = {"a": "ə", "e": "ə", "i": "ə", "o": "ə", "u": "ə"}
_PLAIN_Y = {"y": "iː"}

_SYMBOLS = sorted(_VOWELS + _CONSONANTS, key=lambda kv: -len(kv[0]))
_VOWEL_SET = {ipa for _, ipa in _VOWELS} | {"ə", "iː"}

_PRIMARY = '"'
_SECONDARY = "`"

_DIACRITICS = "āăâäȧēĕẽīĭōŏôūŭûụṳṉ"


def to_ipa(raw: str) -> str | None:
    """'(klŏk"līk`)' -> 'ˈklɒkˌlaɪk'. Returns None on anything this module
    can't safely convert (see module docstring's "Fails closed" section)."""
    text = raw.strip()
    if not text.startswith("(") or ")" not in text:
        return None
    text = text[text.index("(") + 1:text.rindex(")")]
    text = text.split(";")[0].strip()  # drop a trailing foreign-variant clause
    text = re.split(r"\s+or\s+", text)[0].strip()  # take the first of several listed alternates
    if not text:
        return None
    if "?" in text or "#" in text or "N" in text:
        return None
    if text.startswith("-"):
        return None
    if re.fullmatch(r"[A-Za-z]+", text) and not any(c in text for c in _DIACRITICS):
        return None

    out: list[str] = []
    pending_start = 0
    last_syll_start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "*-":
            pending_start = len(out)
            i += 1
            continue
        if ch in (_PRIMARY, _SECONDARY):
            mark = "ˈ" if ch == _PRIMARY else "ˌ"
            out.insert(last_syll_start, mark)
            pending_start = len(out)
            i += 1
            continue
        if ch == "'":
            out.append("ə")
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
        if matched:
            continue
        if ch in _PLAIN_Y:
            out.append(_PLAIN_Y[ch])
            last_syll_start = pending_start
            pending_start = len(out)
            i += 1
            continue
        if ch in _PLAIN_VOWEL_SCHWA:
            out.append(_PLAIN_VOWEL_SCHWA[ch])
            last_syll_start = pending_start
            pending_start = len(out)
            i += 1
            continue
        return None
    return "".join(out) if out else None
