"""Local Wiktionary dictionary — vocab.wiktionary, ~500k terms already loaded
in Postgres (see the README's Postgres section). No network round-trip, so
it's tried before any of the online sources in dictionary.py, and it's
checked first in the validity gate too: this dump was built with no
"Proper noun" POS category at all, so membership alone is a much cleaner
"this is a real word, not a name" signal than the frequency-based
authorities (SymSpell/WordNet/wordfreq) validity.py otherwise relies on —
those are all polluted by real names that happen to have some web
frequency (see the proper-noun audit).
"""

from __future__ import annotations

from .model import Candidate, normalize_pos

_COARSE_POS = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb"}

Entry = tuple[str, str, str, str, bool, bool]  # pos, definition, ipa, etymology, is_archaic, is_obsolete


def build_lexicon(conn, lemmas: set[str], schema: str = "vocab") -> dict[str, list[Entry]]:
    """One bulk query for every candidate lemma at once (empty dict if `lemmas`
    is empty — avoids an `= ANY('{}')` query that would just scan nothing)."""
    if not lemmas:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT lower(term), part_of_speech, definition, us_pronunciation,
                       etymology, is_archaic, is_obsolete
                FROM {schema}.wiktionary WHERE lower(term) = ANY(%s)""",
            (list(lemmas),),
        )
        rows = cur.fetchall()
    lexicon: dict[str, list[Entry]] = {}
    for term, pos, definition, ipa, etymology, is_archaic, is_obsolete in rows:
        lexicon.setdefault(term, []).append(
            (pos or "", definition or "", ipa or "", etymology or "", bool(is_archaic), bool(is_obsolete))
        )
    return lexicon


def lookup_one(conn, lemma: str, schema: str = "vocab") -> list[Entry]:
    """Single-word convenience wrapper for call sites without a pre-built
    lexicon (the webapp's rescue path, refill/deepen's CSV backfills)."""
    return build_lexicon(conn, {lemma.lower()}, schema).get(lemma.lower(), [])


def _pick_entry(cand: Candidate, entries: list[Entry]) -> Entry:
    """Mirrors dictionary._pick_sense's POS-matching preference."""
    if len(entries) == 1:
        return entries[0]
    tagged = _COARSE_POS.get(cand.pos)
    if tagged:
        for e in entries:
            if e[0].lower() == tagged:
                return e
    return entries[0]


def enrich(cand: Candidate, lexicon: dict[str, list[Entry]]) -> bool:
    """Fill definition/POS/IPA/etymology from the local dictionary. No
    synonyms column in this dump — a word resolved here just won't have
    synonyms, which is an acceptable tradeoff for skipping the network
    entirely. Returns False (leaving cand untouched) on a miss so the caller
    can fall back to dictionary.enrich()."""
    entries = lexicon.get(cand.lemma.lower())
    if not entries:
        return False
    pos, definition, ipa, etymology, _is_archaic, _is_obsolete = _pick_entry(cand, entries)
    cand.part_of_speech = normalize_pos(pos)
    cand.definition = definition.split(";")[0].strip()  # first (primary) sense
    if ipa:
        cand.ipa = ipa
    if etymology:
        cand.etymology = etymology
    cand.definition_source = "Local Wiktionary (DB)"
    return True
