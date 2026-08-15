"""Unified definition-acquisition cascade (§ maintenance redesign).

One place that defines "the cascade order," used consistently by ingest
(pipeline.py), refill_definitions/deepen_definitions (db.py), and the
stand-alone scripts/lookup_word.py -- replacing four previously separate,
drifting implementations of overlapping cascades that had no single source
of truth for tier order, rate-limiting, or POS handling.

Tier order, cheapest/most-reliable first:
  1. LOCAL    -- local Wiktionary dump (localdict.py). No network, no rate
                 limit, and structurally free of "Proper noun" POS entries.
  2. FREE     -- Free Dictionary API + Wiktionary REST (dictionary.py). No
                 key, no rate limit (dictionary._get already backs off on
                 429/5xx on its own).
  3. OED      -- the OED reference dataset, OCR'd from scanned volumes into
                 its own `oed` schema (oed/definitions.py). No network, no
                 rate limit -- but unlike every tier above, its text was
                 never editorially curated for this purpose: oed.definition
                 rows come from a sense-splitter its own module docstring
                 calls "a first pass, not a calibrated parser," and can
                 still contain a truncated etymology fragment or (rarely) a
                 sense bled in from an adjacent headword's entry.
                 oed/definitions.py filters out the failure patterns it can
                 detect (an unbalanced bracket, an OED "(See quot...)"
                 cross-reference with no standalone prose), but what
                 survives isn't given the same trust as a curated
                 dictionary hit -- see the variant_flag_reason note in
                 _from_oed. Placed after FREE (a clean modern gloss wins by
                 default when one exists) and before MW/WORDNIK (costs
                 nothing, so there's no reason to burn either's rate-limited
                 quota first when OED might already have it) -- Brian's
                 call, made with the measured hit-rate in hand.
  4. MW       -- Merriam-Webster Collegiate API (mw.py). Needs
                 MW_DICTIONARY_API_KEY; capped at 1000 requests/day (mw.py's
                 own on-disk cache + quota counter), no per-request pacing
                 needed unlike Wordnik, so it's cheaper per word despite the
                 daily ceiling -- placed ahead of WORDNIK for that reason.
                 Was previously only reachable via pipeline.py's own
                 hand-rolled ingest-time step (LOCAL -> FREE -> MW); folded
                 in here so every caller gets it, not just ingest. Uniquely
                 among tiers, a hit can also set cand.verdict/reject_reason
                 directly: MW's own "<Language> noun/verb/..." POS
                 convention (mw.is_foreign_pos) is real, load-bearing
                 evidence a headword is a not-yet-naturalized loanword
                 (confirmed live: "hatari" -> "Swahili noun") that no other
                 tier can detect, so it casts the word out right here
                 instead of silently defining it as if it were English.
  5. WORDNIK  -- Century/GCIDE/AHD (deepdef.py). Needs WORDNIK_API_KEY;
                 capped at 5 requests/minute on the free tier -- paced by
                 THIS module (see _pace_wordnik), not by the caller, so
                 every caller gets correct pacing automatically instead of
                 each one re-implementing (and, historically, over-
                 applying) its own blanket per-word delay regardless of
                 whether Wordnik was even reached that word.
  6. YOURDICT -- yourdictionary.com (deepdef.py). Scraped, keyless, no cap.
  7. WEB      -- web search + local LLM extraction (websearch.py). The true
                 last resort: reads real search-result snippets and has the
                 model extract a definition that is actually present in
                 them -- it never invents one.

Free/keyless tiers go first everywhere now, a deliberate change from
scripts/lookup_word.py's original Wordnik-first order: it protects
Wordnik's tight 5-req/min budget for words nothing free can resolve. Cost:
a word both a free source and Wordnik could define now gets the free
source's (often blander, modern) gloss instead of Wordnik's archaic one --
an accepted tradeoff for not burning rate-limit budget on words that don't
need it.

`max_tier` lets a caller cap how deep the cascade goes without duplicating
tier-selection logic -- e.g. ingest/refill want cheap tiers only
(max_tier=Tier.FREE), deepen/lookup_word.py want full depth (Tier.WEB).

POS-repair: after a hit at ANY tier, if part_of_speech is still blank -- a
real, confirmed gap in dictionary.py's extraction (the source API's own
partOfSpeech field can be blank for the winning sense) and a structural one
in websearch.py/yourdictionary's scrape (neither has a POS to give at all)
-- borrow it from the already-loaded local lexicon if the lemma is there,
without touching whichever tier's definition text actually won. Costs
nothing extra: the lexicon is already in memory for Tier 1.
"""

from __future__ import annotations

import time
from enum import IntEnum

from . import deepdef, dictionary, localdict, mw, websearch
from .localdict import _pick_entry
from .model import Candidate, RejectReason, Verdict, normalize_pos
from .oed import definitions as oed_definitions
from .oed.lemma import pos_categories as oed_pos_categories

_WORDNIK_MIN_INTERVAL = 12.5  # seconds; free tier caps at 5 requests/minute


class Tier(IntEnum):
    LOCAL = 1
    FREE = 2
    OED = 3
    MW = 4
    WORDNIK = 5
    YOURDICT = 6
    WEB = 7


_last_wordnik_call: float = 0.0


def _pace_wordnik() -> None:
    """Block just long enough to respect the 5-req/min free-tier cap. A
    module-level timestamp, not per-caller state, so pacing is correct
    regardless of how many different call sites reach this tier."""
    global _last_wordnik_call
    wait = _WORDNIK_MIN_INTERVAL - (time.monotonic() - _last_wordnik_call)
    if wait > 0:
        time.sleep(wait)
    _last_wordnik_call = time.monotonic()


def _from_mw(cand: Candidate, session, api_key: str) -> bool:
    """MW tier: exact-headword-matched entries only (mw.exact_matches --
    MW's API does fuzzy full-text search and will happily return an
    unrelated idiom for a non-headword query), homograph picked by the
    tagger's coarse POS (mw.pick_entry). A foreign-language loanword,
    caught via MW's raw (pre-normalize_pos) part_of_speech string, still
    fills in the definition (for context) but also marks the candidate
    DROP/FOREIGN_LANGUAGE -- the caller doesn't need to special-case this,
    just check cand.verdict same as any other cast-out path."""
    entries = mw.exact_matches(mw.lookup_api(cand.lemma, api_key, session), cand.lemma)
    if not entries:
        return False
    entry = mw.pick_entry(entries, cand.pos)
    cand.definition = "; ".join(entry.definitions)
    cand.definition_source = entry.source
    cand.part_of_speech = normalize_pos(entry.part_of_speech)
    if entry.etymology:
        cand.etymology = entry.etymology
    # NOT cand.ipa: MW's pronunciation is a proprietary respelling, not real
    # IPA -- word.ipa is trusted elsewhere (audio.py's Azure TTS synthesis)
    # to actually contain IPA.
    if mw.is_foreign_pos(entry.part_of_speech):
        cand.verdict = Verdict.DROP
        cand.reject_reason = RejectReason.FOREIGN_LANGUAGE
        cand.interesting_reason = (
            f"Merriam-Webster listed this as a {entry.part_of_speech.lower()} — cast out")
    return True


_OED_REVIEW_FLAG = "oed_unverified"
_OED_REVIEW_NOTE = "definition sourced from OED; sense-splitting isn't fully reliable yet, not human-reviewed"


def _from_oed(cand: Candidate, oed_lexicon: dict) -> bool:
    """OED tier: senses matched by exact lemma headword, homograph picked by
    the tagger's coarse POS (oed.definitions.pick_sense, same pattern as
    localdict._pick_entry/mw.pick_entry). Unlike every other tier, a hit
    here also marks the candidate for human review (variant_flag_reason,
    the same "probably fine, human should glance" channel
    validity_score.variant_reject_reason already uses) rather than being
    trusted silently -- oed/definitions.py's filter catches the failure
    patterns it can detect, but this is real OCR'd dictionary text that was
    never curated for this purpose, and this project's other definition
    sources all are. Doesn't overwrite an existing flag from an earlier
    stage (pipeline.py's own variant_reject_reason check runs after
    enrichment and is a stronger, more specific signal when it fires)."""
    senses = oed_lexicon.get(cand.lemma.lower())
    if not senses:
        return False
    sense = oed_definitions.pick_sense(cand.pos, senses)
    cand.definition = sense.definition
    cand.definition_source = "OED"
    categories = oed_pos_categories(sense.part_of_speech)
    if categories:
        cand.part_of_speech = normalize_pos(cand.pos if cand.pos in categories else sorted(categories)[0])
    if sense.etymology:
        cand.etymology = sense.etymology
    if not cand.variant_flag_reason:
        cand.variant_flag_reason = _OED_REVIEW_FLAG
        cand.variant_flag_note = _OED_REVIEW_NOTE
    return True


def resolve_definition(
    cand: Candidate,
    *,
    max_tier: Tier = Tier.WEB,
    lexicon: dict | None = None,
    oed_lexicon: dict | None = None,
    session=None,
    wordnik_key: str | None = None,
    mw_api_key: str | None = None,
    llm=None,
) -> Tier | None:
    """Try tiers in order up to max_tier, stopping at the first hit. Mutates
    `cand` in place (definition/definition_source/part_of_speech/ipa/
    etymology/synonyms, whichever fields that tier's source provides).
    Returns the Tier that resolved it, or None if every tier up to
    max_tier missed. `lexicon` (from localdict.build_lexicon) and `session`
    (from dictionary.make_session) are expected to be built once per batch
    by the caller and passed in -- omitting `lexicon` simply skips Tier
    LOCAL (e.g. scripts/lookup_word.py, which has no database at all).
    `oed_lexicon` (from oed.definitions.definition_lexicon) works the same
    way for Tier OED -- omitting it just skips that tier."""
    lexicon = lexicon or {}
    oed_lexicon = oed_lexicon or {}
    resolved: Tier | None = None

    if localdict.enrich(cand, lexicon):
        resolved = Tier.LOCAL

    if resolved is None and max_tier >= Tier.FREE:
        session = session or dictionary.make_session()
        dictionary.enrich(cand, session)
        if cand.definition:
            resolved = Tier.FREE

    if resolved is None and max_tier >= Tier.OED:
        if _from_oed(cand, oed_lexicon):
            resolved = Tier.OED

    if resolved is None and max_tier >= Tier.MW:
        key = mw_api_key if mw_api_key is not None else mw.mw_api_key()
        if key:
            session = session or dictionary.make_session()
            if _from_mw(cand, session, key):
                resolved = Tier.MW

    if resolved is None and max_tier >= Tier.WORDNIK:
        key = wordnik_key if wordnik_key is not None else deepdef.wordnik_key()
        if key:
            session = session or dictionary.make_session()
            _pace_wordnik()
            if deepdef._from_wordnik(cand, session, key):
                resolved = Tier.WORDNIK

    if resolved is None and max_tier >= Tier.YOURDICT:
        session = session or dictionary.make_session()
        if deepdef._from_yourdictionary(cand, session):
            resolved = Tier.YOURDICT

    if resolved is None and max_tier >= Tier.WEB and llm is not None:
        if websearch.define_via_web(cand, llm):
            resolved = Tier.WEB

    if resolved is not None:
        apply_pos_repair(cand, lexicon)

    return resolved


def apply_pos_repair(cand: Candidate, lexicon: dict | None) -> None:
    """The POS-repair sub-step on its own, for a call site that resolves a
    tier outside resolve_definition's own cascade (deepen_definitions' web
    tier, gated on a validity_score check resolve_definition itself has no
    concept of) but still wants the same lexicon-borrow behavior every other
    caller gets automatically. No-op if part_of_speech is already set."""
    if cand.part_of_speech:
        return
    entries = (lexicon or {}).get(cand.lemma.lower())
    if entries:
        pos = _pick_entry(cand, entries)[0]
        if pos:
            cand.part_of_speech = normalize_pos(pos)
