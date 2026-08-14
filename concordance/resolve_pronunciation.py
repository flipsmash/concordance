"""Unified pronunciation-acquisition cascade (§ audio pronunciation).

Feeds word.ipa, the one field audio.py trusts for TTS synthesis (Azure
SSML `<phoneme>`) and for matching against Commons recordings. Mirrors
resolve.py's shape for definitions: one place that defines tier order and
the per-word decision, replacing what used to be two separately-maintained
implementations (compute_ipa's inline `kaikki_ipa or wn_ipa or
local_wiktionary_ipa` chain, and backfill_ipa_from_oed's own standalone
priority/candidate-selection logic) with no single source of truth between
them for tier order.

Tier order below preserves exactly what compute_ipa/backfill_ipa_from_oed
already did before this module existed (kaikki wins when present, oed only
ever fills a gap nothing else could) -- this module unifies WHERE that
order lives, not what it is. Unlike resolve.py's definition cascade, every
tier here is a LOOKUP against data the caller already fetched/loaded in
bulk (see compute_ipa) -- this module makes no live per-word network call
itself. The one genuinely expensive/rate-limited source (Wordnik) already
paid its network cost in a separate prerequisite stage,
`fetch_wordnik_pronunciations` / `concordance wordnik-pron`, precisely
because its 5-req/min cap makes a live call from inside a per-word cascade
unworkable -- that stage stays separate, this module just decides what to
do with whatever it already fetched.

  1. KAIKKI            -- kaikki/Wiktextract dump (wiktextract.py). Built
                           once per batch from a local gzipped dump, no
                           network at lookup time.
  2. WORDNIK            -- word.wordnik_pron_raw/wordnik_pron_type, already
                           fetched by `wordnik-pron`, converted via the
                           matching notation converter (ahd.py/arpabet.py;
                           gcide-diacritical has no converter yet).
  3. LOCAL_WIKTIONARY   -- vocab.wiktionary's us_pronunciation column
                           (localdict.py). The same underlying Wiktionary
                           data kaikki's dump draws from, just a different
                           snapshot -- low yield on its own, but free (the
                           DB connection is already open), so it still gets
                           a turn.
  4. OED                -- the oed schema's double-pass vision-LLM-verified
                           pronunciation_ipa (oed/pronunciation.py). Highest-
                           confidence source (human-verified twice over),
                           but placed LAST and never used to override an
                           earlier tier's answer (see compute_ipa's
                           only_missing gate) -- upgrading a lower-confidence
                           existing IPA to OED's is a deliberate non-goal,
                           and coverage is still partial (grows with each
                           future `oed-ingest` run), so it's positioned as a
                           gap-filler, not promoted ahead of sources that
                           already cover most of the corpus. Only usable
                           when every oed.entry homograph for a headword
                           agrees (OED's part_of_speech is unparsed OCR
                           soup, not usable to pick the "right" homograph)
                           -- a genuine conflict contributes nothing, not a
                           guess.

Every candidate is sanity-checked with audio.looks_like_english_ipa before
it's allowed to win a tier -- a source producing garbage (or a
wrong-language transcription) doesn't get to poison word.ipa just because
it technically had *something* to say.
"""

from __future__ import annotations

from enum import IntEnum

from . import ahd, arpabet, audio, wiktextract


class Tier(IntEnum):
    KAIKKI = 1
    WORDNIK = 2
    LOCAL_WIKTIONARY = 3
    OED = 4


def _wordnik_ipa(raw: str | None, rtype: str | None) -> str | None:
    if not raw:
        return None
    if rtype == "IPA":
        converted = raw
    elif rtype == "arpabet":
        converted = arpabet.to_ipa(raw)
    elif rtype == "ahd-5":
        converted = ahd.to_ipa(raw)
    else:
        return None  # gcide-diacritical: no converter yet
    return converted if converted and audio.looks_like_english_ipa(converted) else None


def _local_wiktionary_ipa(local_entries: list[tuple] | None) -> str | None:
    for _pos, _definition, ipa, *_rest in (local_entries or []):
        if ipa and audio.looks_like_english_ipa(ipa):
            return ipa
    return None


def _oed_ipa(oed_matches: list[str] | None) -> str | None:
    """None for both "no oed data" and "homographs disagree" -- the caller
    doesn't need to tell those apart, both mean this tier has nothing safe
    to contribute for this headword."""
    if not oed_matches or len(oed_matches) > 1:
        return None
    candidate = oed_matches[0]
    return candidate if audio.looks_like_english_ipa(candidate) else None


def resolve_ipa(
    *,
    kaikki_entry: dict | None = None,
    wordnik_raw: str | None = None,
    wordnik_type: str | None = None,
    local_entries: list[tuple] | None = None,
    oed_matches: list[str] | None = None,
    max_tier: Tier = Tier.OED,
) -> tuple[str, Tier] | None:
    """Try tiers in order up to max_tier, returning (ipa, tier) for the
    first one with a valid candidate, or None if every tier up to max_tier
    missed. Every argument is a per-word slice of a lexicon the caller
    already built in bulk for the whole batch -- this function does no I/O
    of its own, so it's cheap to call for every candidate in a loop."""
    if max_tier >= Tier.KAIKKI:
        kaikki_ipa = wiktextract.best_ipa((kaikki_entry or {}).get("ipa", []))
        if kaikki_ipa and audio.looks_like_english_ipa(kaikki_ipa):
            return kaikki_ipa, Tier.KAIKKI

    if max_tier >= Tier.WORDNIK:
        wn_ipa = _wordnik_ipa(wordnik_raw, wordnik_type)
        if wn_ipa:
            return wn_ipa, Tier.WORDNIK

    if max_tier >= Tier.LOCAL_WIKTIONARY:
        lw_ipa = _local_wiktionary_ipa(local_entries)
        if lw_ipa:
            return lw_ipa, Tier.LOCAL_WIKTIONARY

    if max_tier >= Tier.OED:
        oed_ipa = _oed_ipa(oed_matches)
        if oed_ipa:
            return oed_ipa, Tier.OED

    return None
