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

import re
import unicodedata
from typing import Callable

from .model import Candidate, normalize_pos

_COARSE_POS = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb"}

Entry = tuple[str, str, str, str, bool, bool]  # pos, definition, ipa, etymology, is_archaic, is_obsolete

# The dump often glosses a term as a bare "abbreviation of X." -- true for
# real clippings (contemp -> contemporary) but just as often mislabels a
# plain spelling/case/compound-spacing variant (vizor -> visor, waterpower
# -> water power) or, worse, a claim that doesn't even hold up under the
# label (waterfit -> "abbreviation of aquafitness", which share almost no
# letters). The dump gives no way to tell those apart from the label alone,
# so resolve_stub_definition() ignores the label and resolves X's own real
# definition instead, choosing an honest relation word by shape.
_STUB_RE = re.compile(r"^abbreviation of\s+(.+?)\.?$", re.IGNORECASE)


def _is_stub(definition: str) -> bool:
    return bool(_STUB_RE.match((definition or "").strip()))


def _fold_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _target_candidates(target_raw: str) -> list[str]:
    """Normalized lookup keys to try for a stub's target phrase: the full
    phrase, and (only for a genuinely two-word compound, e.g. "water
    power" for waterpower) the phrase with the space removed -- each also
    tried with accents folded off (calèche/caleche). Deliberately does NOT
    fall back to just the target's first word for a longer phrase: tried
    that for multi-word/technical targets (e.g. "hoi polloi", "polyclonal
    immunoglobulin G") and it found a real but UNRELATED word ("hoi" ->
    "hey!", "polyclonal" -> a different, more general sense) often enough
    to be worse than leaving the word unresolved -- a confidently wrong
    substitution is exactly the class of bug this is fixing, not a
    tolerable trade for higher coverage."""
    keys: list[str] = []
    words = target_raw.split(" ")
    candidates = [target_raw] if len(words) != 2 else [target_raw, target_raw.replace(" ", "")]
    for c in candidates:
        cl = c.lower()
        if cl and cl not in keys:
            keys.append(cl)
        folded = _fold_accents(cl)
        if folded and folded not in keys:
            keys.append(folded)
    return keys


def _best_entry(entries: list[Entry] | None) -> Entry | None:
    """Prefer a substantive definition over a bare stub among same-term entries."""
    if not entries:
        return None
    for e in entries:
        if not _is_stub(e[1]):
            return e
    return entries[0]


_MARKUP_RESIDUE_RE = re.compile(r"\[\[|\{\{|\}\}|\|\w+=")


def _resolve_real_gloss(
    term: str, definition: str, lookup: Callable[[str], Entry | None], _depth: int = 0
) -> tuple[str, str] | None:
    """Follow a chain of stub definitions down to the first substantive
    one (e.g. acronychal -> acronycal -> acronical). Returns
    (resolved_term, resolved_gloss), or None if nothing resolvable was
    found -- the caller should flag the word for human review rather than
    guess. Bails (returns None) on any leftover wiki-template markup
    ("[[w:...", "(|short=yes}})") in EITHER the stub or the candidate's own
    gloss -- corrupted source text either way, not worth guessing at or
    passing through to a user."""
    if _depth > 5:
        return None
    definition = (definition or "").strip()
    if not definition or _MARKUP_RESIDUE_RE.search(definition):
        return None
    m = _STUB_RE.match(definition)
    if not m:
        return (term, definition)
    target_raw = re.sub(r"\s*\([^)]*\)\s*$", "", m.group(1).strip())
    if not target_raw:
        return None
    for key in _target_candidates(target_raw):
        entry = lookup(key)
        if not entry:
            continue
        entry_def = (entry[1] or "").split(";")[0].strip()
        deeper = _resolve_real_gloss(key, entry_def, lookup, _depth + 1)
        if deeper:
            return deeper
    return None


def resolve_stub_definition(headword: str, definition: str, lookup: Callable[[str], Entry | None]) -> str | None:
    """If `definition` is a bare "abbreviation of X[.]" stub, resolve X's
    real definition (via `lookup(term) -> Entry | None`, injected so this
    works against either a pre-built in-ingest lexicon or a live DB) and
    render an honest, content-bearing replacement. Returns None if `definition`
    isn't a stub, or if no real content for X could be found."""
    m = _STUB_RE.match((definition or "").strip())
    if not m:
        return None
    target_raw = re.sub(r"\s*\([^)]*\)\s*$", "", m.group(1).strip())
    if not target_raw:
        return None
    resolved = _resolve_real_gloss(headword, definition, lookup)
    if resolved is None:
        return None
    _, gloss = resolved
    if not gloss:
        return None
    # A length gap of 1-2 characters (chiel/chield, gramary/gramarye,
    # ratlin/ratline) is a trailing-letter spelling variant, not a real
    # clipping -- verified against the live backlog: every case with a gap
    # of >=3 (contemp/contemporary, distr/distribution, supe/superintendent)
    # reads as a genuine abbreviation, and every gap of 1-2 does not.
    first_word = target_raw.split(" ", 1)[0]
    is_abbrev = (
        " " not in target_raw
        and first_word.lower().startswith(headword.lower())
        and len(first_word) - len(headword) >= 3
    )
    relation = "Abbreviation" if is_abbrev else "Variant spelling"
    return f"{relation} of {target_raw} — {gloss.rstrip('.')}."


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
    """Mirrors dictionary._pick_sense's POS-matching preference. Among
    ties, prefers a substantive definition over a bare "abbreviation of X"
    stub -- the dump sometimes carries both under the same term (e.g. a
    real "Relating to a Spanish baroque..." entry for "churrigueresque"
    alongside a redundant case-variant stub "abbreviation of
    Churrigueresque."), and picking the stub first is pure information loss."""
    if len(entries) == 1:
        return entries[0]
    tagged = _COARSE_POS.get(cand.pos)
    pool = entries
    if tagged:
        pos_matches = [e for e in entries if e[0].lower() == tagged]
        if pos_matches:
            pool = pos_matches
    return _best_entry(pool) or pool[0]


def stub_target_candidates(definition: str) -> list[str]:
    """Public wrapper: the lookup keys resolve_stub_definition would try
    for a given "abbreviation of X" definition, or [] if it isn't a stub.
    For callers (e.g. a DB backfill) whose stub headwords are known only
    from stored text, not from a lexicon already covering them -- so
    expand_lexicon_for_stubs's own scan (which only sees words already in
    the lexicon) would miss them entirely."""
    m = _STUB_RE.match((definition or "").strip())
    if not m:
        return []
    target_raw = re.sub(r"\s*\([^)]*\)\s*$", "", m.group(1).strip())
    return _target_candidates(target_raw) if target_raw else []


def expand_lexicon_for_stubs(conn, lexicon: dict[str, list[Entry]], schema: str = "vocab") -> None:
    """Mutates `lexicon` in place: for every "abbreviation of X" stub
    definition already in it, bulk-fetches X (and, if X is itself a stub,
    follows the chain a few rounds) so enrich() can resolve stubs to real
    content purely from the in-memory lexicon -- no live per-word DB call
    needed during (possibly parallel) enrichment. Call once, right after
    build_lexicon, while `conn` is still open."""
    for _round in range(3):
        needed: set[str] = set()
        for entries in list(lexicon.values()):
            for _pos, definition, *_rest in entries:
                m = _STUB_RE.match((definition or "").split(";")[0].strip())
                if not m:
                    continue
                target_raw = re.sub(r"\s*\([^)]*\)\s*$", "", m.group(1).strip())
                if not target_raw:
                    continue
                needed.update(k for k in _target_candidates(target_raw) if k not in lexicon)
        if not needed:
            return
        more = build_lexicon(conn, needed, schema)
        if not more:
            return
        for k, v in more.items():
            lexicon.setdefault(k, v)


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
    definition = definition.split(";")[0].strip()  # first (primary) sense
    if _is_stub(definition):
        resolved = resolve_stub_definition(cand.lemma, definition, lambda k: _best_entry(lexicon.get(k)))
        if resolved:
            definition = resolved
        else:
            cand.variant_flag_reason = cand.variant_flag_reason or "abbreviation_stub"
            cand.variant_flag_note = cand.variant_flag_note or (
                f'Local Wiktionary only has "{definition}" -- target not resolvable in the local dump.'
            )
    cand.definition = definition
    if ipa:
        cand.ipa = ipa
    if etymology:
        cand.etymology = etymology
    cand.definition_source = "Local Wiktionary (DB)"
    return True
