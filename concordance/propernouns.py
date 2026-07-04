"""Stage 5 — strip proper nouns (§04).

Layered, cheapest test first:
  1. the tagger — spaCy called it PROPN or put it inside a named entity;
  2. in-book capitalization ratio — catches invented names (Aragorn, Winterfell)
     that no tagger knows, by how consistently the word is capitalized in
     mid-sentence position across the whole book.
The model backstop (an explicit "reject names" instruction) lives in the judge.

The hard case — names that collide with real words (Baker, Rose, Mark) — is why
the capitalization ratio matters: it separates "the baker" from "Mr. Baker"
better than any single dictionary check.
"""

from __future__ import annotations

from .config import Config
from .model import Candidate, RejectReason, Verdict


def strip_proper_nouns(candidates: dict[str, Candidate], cfg: Config) -> None:
    for cand in candidates.values():
        if cand.verdict is not None:
            continue
        # Real mid-sentence capitalization is a strong, standalone signal.
        capitalization_says_name = cand.cap_ratio >= cfg.cap_ratio_threshold
        # The tagger alone is unreliable on a lone sentence-initial token
        # (where every word is capitalized). A single-occurrence PROPN always
        # has ratio 1.0, so ratio can't corroborate — recurrence must. This
        # keeps one-off words like "Motes of dust…" out of the proper-noun
        # bucket; a true one-off invented name still gets dropped downstream by
        # the validity gate (unattested, no dictionary vouch).
        tagger = cand.propn_ratio >= 0.5 or cand.ent_ratio >= 0.5
        tagger_corroborated = tagger and cand.count >= 2
        if capitalization_says_name or tagger_corroborated:
            cand.verdict = Verdict.DROP
            cand.reject_reason = RejectReason.PROPER_NOUN
