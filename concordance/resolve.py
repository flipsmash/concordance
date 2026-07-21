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
  3. WORDNIK  -- Century/GCIDE/AHD (deepdef.py). Needs WORDNIK_API_KEY;
                 capped at 5 requests/minute on the free tier -- paced by
                 THIS module (see _pace_wordnik), not by the caller, so
                 every caller gets correct pacing automatically instead of
                 each one re-implementing (and, historically, over-
                 applying) its own blanket per-word delay regardless of
                 whether Wordnik was even reached that word.
  4. YOURDICT -- yourdictionary.com (deepdef.py). Scraped, keyless, no cap.
  5. WEB      -- web search + local LLM extraction (websearch.py). The true
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

from . import deepdef, dictionary, localdict, websearch
from .localdict import _pick_entry
from .model import Candidate, normalize_pos

_WORDNIK_MIN_INTERVAL = 12.5  # seconds; free tier caps at 5 requests/minute


class Tier(IntEnum):
    LOCAL = 1
    FREE = 2
    WORDNIK = 3
    YOURDICT = 4
    WEB = 5


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


def resolve_definition(
    cand: Candidate,
    *,
    max_tier: Tier = Tier.WEB,
    try_free: bool = True,
    lexicon: dict | None = None,
    session=None,
    wordnik_key: str | None = None,
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

    `try_free=False` skips Tier FREE specifically while still trying tiers
    above it -- deepen_definitions's one real need: by the time a word
    reaches deepen in the normal maintain sequence, refill_definitions has
    already tried Free Dictionary/Wiktionary on that exact lemma and failed
    (deterministically, same sources), so retrying it would just be more of
    the same wasted redundancy Tier LOCAL already had before this module
    existed -- not a `max_tier` ceiling, since deepen still wants WORDNIK/
    YOURDICT/WEB above it."""
    lexicon = lexicon or {}
    resolved: Tier | None = None

    if localdict.enrich(cand, lexicon):
        resolved = Tier.LOCAL

    if resolved is None and try_free and max_tier >= Tier.FREE:
        session = session or dictionary.make_session()
        dictionary.enrich(cand, session)
        if cand.definition:
            resolved = Tier.FREE

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
