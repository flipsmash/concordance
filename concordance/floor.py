"""Stage 4 — the frequency floor.

A cheap stop-word-style cut: drop lemmas common enough that they can't be
"interesting". This is a floor ONLY. There is deliberately no rarity ceiling —
mid-frequency words are exactly where the good vocabulary lives, so rarity is
never used to rule a word *in* or to discard it for being obscure.
"""

from __future__ import annotations

from .config import Config
from .model import Candidate, RejectReason, Verdict
from .validity_score import effective_zipf


def apply_floor(candidates: dict[str, Candidate], cfg: Config) -> None:
    """Annotate each candidate with its Zipf frequency and drop the common ones.

    Uses effective_zipf (root-aware) rather than the lemma's own Zipf alone,
    so a transparent derivative of a common word (unbuttoned, quickly) floors
    out the same as its root would, instead of leaking through because
    wordfreq undercounts the derived form itself."""
    for cand in candidates.values():
        if cand.verdict is not None:
            continue
        cand.zipf = effective_zipf(cand.lemma)
        if cand.zipf >= cfg.min_zipf:
            cand.verdict = Verdict.DROP
            cand.reject_reason = RejectReason.FREQUENCY_FLOOR
