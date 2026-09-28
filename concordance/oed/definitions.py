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

_MIN_LENGTH = 4
_MAX_LENGTH = 600

_SEE_QUOT_RE = re.compile(r"^\(see quot[^)]*\)\s*", re.IGNORECASE)
# Citation year, with OED's optional circa/ante letter glued directly onto
# the digits (c1400, A1450) -- no \w/\w boundary between the letter and the
# digit, so the boundary has to anchor before the optional letter, not the
# digit (matches parse.py's own _YEAR_RE intent, but case-insensitive: the
# original is lowercase-only and misses a sentence-leading "C1400").
_YEAR_CUT_RE = re.compile(r"\b[ac]?1[0-9]\d{2}\b", re.IGNORECASE)   # OED cites from the 1000s on


@dataclass
class OedSense:
    entry_id: int
    part_of_speech: str  # OED's raw abbreviation string, e.g. "v", "a sb"
    etymology: str
    definition: str  # cleaned, semicolon-joined across this entry's senses


# --- the entry header that precedes the gloss ---------------------------------
#
# A sense's text often still opens with the entry's header material, which
# the OCR'd layout ends with a period and TWO spaces before the gloss:
#   "Obs. Also abaue, abaw(e.  trans. To put to confusion, discomfit"
#   "Now Sc. and north, dial. In 4-6 (9) deve, 6 Sc. deiv(e.  fl. intr. To become deaf"
#   "rare.  Violently extreme"          "titju:d).  Want of promptitude"
# Status/region/subject labels, "Also ..." / "In 4-6 ..." / "Forms: ..."
# spelling lists, and the tail of a pronunciation. The header is dropped only
# when every piece of it looks like header material, so a gloss that happens
# to contain ".  " is never cut.
_LABELS = frozenset("""obs rare now sc north south dial nonce nonce-wd nonce-word wd word hist exc
    archaeol archseol arch bot zool med mus law her astron chem anat path eccl theol phys geol min math
    gram rhet mil naut techn local slang colloq vulgar poet fig transf chiefly only and or in the of
    u.s n.e s.w eng scot irish amer anglo-ind cf erron obsol
    sb vbl ppl pple pa adj adv attrib trans intr absol refl pass prop phr prec ellipt spec
    austral austral. canad n.z s.afr n.amer anglo-irish orig collect""".split())
_VARIANTS_INTRO_RE = re.compile(r"^(?:also|forms?:)$", re.IGNORECASE)
# The header ends where 2+ spaces start the gloss.
_HEADER_SPLIT_RE = re.compile(r"\s{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.)])\s+(?=[A-Z(=]|[a-z]+\.)")
# Sense numbering and grammar labels left at the start of the gloss itself.
_LEAD_RE = re.compile(
    r"^(?:(?:[ft]?\s*\d{1,2}(?:\s+\d{1,2})*\s*\.|[ft]l\.|[A-Da-d]\s*\.|[IVX]{1,4}\.|\([a-d]\)|\|\||[*†‡]|"
    r"(?:trans|t\s+rans|intr|in\s+tr|absol|refl|pass|attrib|fig|transf|sb|adj|a|adv|v|vbl|ppl|pa\.\s*pple)\.)\s*)+",
    re.IGNORECASE)
# A citation year the scan garbled (*747, i860, l66o): cut there too.
_OCR_YEAR_CUT_RE = re.compile(r"(?:^|\s)[*il|!2]\d{3}\b|\b01[3-9]\d{2}\b|\b1[3-9]\d[oO]\b")
_TRAILING_SEE_QUOT_RE = re.compile(r"[:;,]?\s*\(?see quot\.?\)?$", re.IGNORECASE)
# A dangling sub-sense marker or sense number where the scan ran on: "... fight. a", "... two. 1"
_TRAILING_MARKER_RE = re.compile(r"[.;,]\s+(?:[a-d]|\d{1,2}|[IVX]{1,4})\s*$")
# A division heading, not a definition: "Literal senses", "Figurative uses"
_HEADING_RE = re.compile(r"^(?:literal|figurative|transferred|general|other|special|simple|compound)\w*"
                         r"\s+(?:senses?|uses?|meanings?)$", re.IGNORECASE)


def _header_like(head: str) -> bool:
    """Every token is a label, a number, OCR noise or a pronunciation tail --
    or comes after "Also"/"Forms:"/"In <number>", which introduce a list of
    old spellings. Any ordinary word before that means it's real prose."""
    from wordfreq import zipf_frequency

    if "[" in head or "]" in head:          # etymology, not header: _plausible's bracket check needs it intact
        return False
    tokens = head.split()
    for i, tok in enumerate(tokens):
        bare = re.sub(r"[^a-z.\-]", "", tok.lower()).strip(".-")
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        # "Also blestly", "Also 8 moys", "Forms: 4-6 ...", "In 4-6 ..." introduce
        # old spellings -- but "Also, too, moreover" (eke) is a real gloss:
        # the next word has to look like a variant, not an everyday word.
        variant_next = bool(nxt) and (any(ch.isdigit() or ch in "(-" for ch in nxt)
                                      or zipf_frequency(nxt.strip(".,;:()").lower(), "en") < 3.0)
        intro = _VARIANTS_INTRO_RE.match(tok.strip(".,;"))
        if (intro and (tok.lower().startswith("form") or variant_next)) or (
                bare == "in" and nxt[:1].isdigit()):
            return True
        if (not bare or bare in _LABELS or len(bare) <= 2 or ")" in tok
                or any(ch.isdigit() for ch in tok) or not re.search(r"[a-z]{3}", tok.lower())):
            continue
        return False
    return bool(tokens)


def _strip_header(text: str) -> str:
    # Prefer the layout's two-space break; fall back to a sentence break for
    # headers the scan closed up ("Also blestly. In a blessed manner").
    for split in (_HEADER_SPLIT_RE, _SENTENCE_SPLIT_RE):
        cut = next((m for m in split.finditer(text[:200]) if _header_like(text[: m.start()])), None)
        if cut:
            text = text[cut.end():]
            break
    prev = None
    while prev != text:                     # "fl. intr. To become deaf" -> "To become deaf"
        prev = text
        text = _LEAD_RE.sub("", text).lstrip()
    return text


def _clean_one_sense(text: str | None) -> str:
    text = (text or "").strip()
    text = _SEE_QUOT_RE.sub("", text).strip()
    for _ in range(3):                      # "Obs. rare. To go": one header piece per pass
        stripped = _strip_header(text)
        if stripped == text:
            break
        text = stripped
    text = _SEE_QUOT_RE.sub("", text).strip()
    for cut in (_YEAR_CUT_RE, _OCR_YEAR_CUT_RE):
        m = cut.search(text)
        if m:
            text = text[: m.start()]
    text = _TRAILING_SEE_QUOT_RE.sub("", text.strip(" .,;:"))
    text = _TRAILING_MARKER_RE.sub("", text)
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
    # Once the entry header is stripped a real gloss can be one word
    # ("Simian", "= LATEWARD"), so length alone no longer separates prose
    # from fragments: require a real word instead, and reject a division
    # heading ("Literal senses") that only introduces the senses below it.
    return (len(text) >= _MIN_LENGTH and text.count("[") == text.count("]")
            and bool(re.search(r"[A-Za-z]{3}", text)) and not _HEADING_RE.match(text)
            and not _header_like(text))         # "Also prae-", "u:mi'nif3r3s)", "Phys. and Path"


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
