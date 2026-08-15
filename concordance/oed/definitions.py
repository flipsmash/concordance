"""Definition lookup against the OED reference dataset -- see resolve.py's
Tier.OED / _from_oed, the single call site this feeds.

oed/parse.py already separates a real OED entry into reliable structured
fields (headword, pronunciation, part_of_speech, etymology all live in their
own oed.entry columns) before splitting the entry's remaining body text into
per-sense rows (oed.definition, ordered by sort_order). That header-parsing
is trustworthy; the sense-splitter is explicitly flagged in its own
docstring as "a first pass, not a calibrated parser," and in practice: senses
after the first are self-contained ~97-100% of the time (measured against
their own entry's raw_text), but 57% of definable entries have only ONE
sense, and that lone sense has a real, if minority, failure mode -- a
truncated fragment with no actual gloss (an unclosed etymology bracket that
swallowed the rest), or OED's "(See quot. 1959.)" convention, where the
definition is only implicit in a citation and there is no standalone prose
to extract at all.

This module works from oed.definition (not a fresh raw_text parse -- an
earlier attempt at that redundantly, and worse, re-implemented what
split_pos/extract_etymology already do) and applies exactly two defenses
against that known failure mode: a minimum length after cleaning, and a
strip of the "(See quot...)" pattern that would otherwise pass through as a
fake definition. This is a filter, not a fix -- the real fix belongs in
oed/parse.py's split_senses, as separate future work.

Each sense's definition_text still carries its own inline dated citations
(oed.quotation is a parallel structured extraction, not a cleaned-up
replacement -- see parse.py's extract_quotations), so every sense is cut at
its first citation year before being joined with the others.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .lemma import pos_categories

DEFAULT_SCHEMA = "oed"

_MIN_LENGTH = 15
_MAX_LENGTH = 600

_SEE_QUOT_RE = re.compile(r"^\(see quot[^)]*\)\s*", re.IGNORECASE)
# Citation year, with OED's optional circa/ante letter glued directly onto
# the digits (c1400, A1450) -- no \w/\w boundary between the letter and the
# digit, so the boundary has to anchor before the optional letter, not the
# digit (matches parse.py's own _YEAR_RE intent, but case-insensitive: the
# original is lowercase-only and misses a sentence-leading "C1400").
_YEAR_CUT_RE = re.compile(r"\b[ac]?1[3-9]\d{2}\b", re.IGNORECASE)


@dataclass
class OedSense:
    entry_id: int
    part_of_speech: str  # OED's raw abbreviation string, e.g. "v", "a sb"
    etymology: str
    definition: str  # cleaned, semicolon-joined across this entry's senses


def _clean_one_sense(text: str | None) -> str:
    text = (text or "").strip()
    text = _SEE_QUOT_RE.sub("", text).strip()
    m = _YEAR_CUT_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip(" .,;:")


def _plausible(text: str) -> bool:
    """Rejects the two known split_senses/extract_etymology failure
    patterns rather than trying to fix them: an unbalanced '[' or ']'
    (etymology-bracket matching failed upstream, so what looks like a sense
    is really an unclosed etymology fragment, or a truncated one missing
    its opening bracket -- "veterinary"/"syphiloma" in the wild), and
    anything too short once "(See quot...)" is stripped (OED's citation-only
    convention, or a genuinely empty sense). Real prose has no legitimate
    reason to contain an unmatched bracket at all, so this stays exact
    equality rather than a looser one-sided check."""
    return len(text) >= _MIN_LENGTH and text.count("[") == text.count("]")


def _pick_definition(parts: list[str]) -> str:
    """First sense (in sort_order) that passes _plausible, not a join of
    everything -- a later sense is where oed.definition's known cross-entry
    bleed shows up (a sense-boundary false match landing on the START of the
    NEXT headword's entry, confirmed live: "lipstick"'s second sense is
    actually the opening of a Hungarian cheese entry). Stopping at the first
    good sense means a clean sense[0] is used as-is and a corrupted tail
    sense is never even looked at."""
    for part in parts:
        cleaned = _clean_one_sense(part)
        if _plausible(cleaned):
            return cleaned[:_MAX_LENGTH].rstrip()
    return ""


def definition_lexicon(conn, headwords: set[str], schema: str = DEFAULT_SCHEMA) -> dict[str, list[OedSense]]:
    """headword_norm -> every distinct lemma=true oed.entry (homographs get
    one OedSense each), definition already cleaned + sense-joined. Degrades
    to {} if the oed schema/entry table doesn't exist yet -- same to_regclass
    pattern oed.db.pronunciation_lexicon uses, since this is now reached from
    the regular ingest/backfill cascade, not just an OED-specific command."""
    if not headwords:
        return {}
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{schema}.entry",))
        if cur.fetchone()[0] is None:
            return {}
        cur.execute(
            f"""SELECT e.id, e.headword_norm, e.part_of_speech, e.etymology, d.definition_text
                FROM {schema}.entry e
                JOIN {schema}.definition d ON d.entry_id = e.id
                WHERE e.lemma AND e.headword_norm = ANY(%s)
                ORDER BY e.id, d.sort_order""",
            (list(headwords),),
        )
        rows = cur.fetchall()

    by_entry: dict[int, dict] = {}
    order: list[int] = []
    for entry_id, headword_norm, pos, etymology, definition_text in rows:
        info = by_entry.get(entry_id)
        if info is None:
            info = by_entry[entry_id] = {
                "headword_norm": headword_norm, "pos": pos or "", "etymology": etymology or "", "parts": [],
            }
            order.append(entry_id)
        info["parts"].append(definition_text)

    lexicon: dict[str, list[OedSense]] = {}
    for entry_id in order:
        info = by_entry[entry_id]
        definition = _pick_definition(info["parts"])
        if not definition:
            continue
        lexicon.setdefault(info["headword_norm"], []).append(
            OedSense(entry_id=entry_id, part_of_speech=info["pos"],
                     etymology=info["etymology"], definition=definition)
        )
    return lexicon


def pick_sense(cand_pos: str, senses: list[OedSense]) -> OedSense:
    """Mirrors localdict._pick_entry/mw.pick_entry: prefer a homograph whose
    OED part_of_speech normalizes to the tagger's own coarse POS, else the
    first (OED's own entry ordering -- not necessarily meaningful, but a
    deterministic tie-break)."""
    if len(senses) == 1:
        return senses[0]
    if cand_pos:
        for s in senses:
            if cand_pos in pos_categories(s.part_of_speech):
                return s
    return senses[0]
