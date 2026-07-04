"""Stage 4 — the frequency floor.

A cheap stop-word-style cut: drop lemmas common enough that they can't be
"interesting". This is a floor ONLY. There is deliberately no rarity ceiling —
mid-frequency words are exactly where the good vocabulary lives, so rarity is
never used to rule a word *in* or to discard it for being obscure.
"""

from __future__ import annotations

from wordfreq import zipf_frequency

from .config import Config
from .model import Candidate, RejectReason, Verdict


def apply_floor(candidates: dict[str, Candidate], cfg: Config) -> None:
    """Annotate each candidate with its Zipf frequency and drop the common ones."""
    for cand in candidates.values():
        if cand.verdict is not None:
            continue
        cand.zipf = zipf_frequency(cand.lemma, "en")
        if cand.zipf >= cfg.min_zipf:
            cand.verdict = Verdict.DROP
            cand.reject_reason = RejectReason.FREQUENCY_FLOOR
