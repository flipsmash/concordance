"""Tunable knobs for a run. Sensible defaults; overridable from the CLI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- Frequency floor (§03.4) -----------------------------------------
    # wordfreq Zipf scale: ~7 = "the", ~4 = "interesting", ~3 = "susurrus",
    # ~1.5 = very rare. Drop anything AT OR ABOVE this as stop-word-like.
    # NOTE: this is a floor only — there is deliberately no rarity ceiling.
    min_zipf: float = 3.5

    # --- Proper-noun stripping (§04) -------------------------------------
    # A lemma capitalized in at least this share of its mid-sentence
    # appearances is treated as a name, even if the tagger missed it.
    cap_ratio_threshold: float = 0.85

    # --- Validity gate (§05) ---------------------------------------------
    # A near-neighbor this many times more frequent (in Zipf terms, i.e.
    # order-of-magnitude jumps) marks the token as a misspelling of it.
    misspelling_zipf_gap: float = 2.0
    # Below this many in-book occurrences, a fully-unvouched token is junk;
    # at or above it, it is treated as a possible coinage -> UNSURE.
    coinage_min_count: int = 3

    # --- LLM judge (§03.8) -----------------------------------------------
    # Default to the 14B — on this hardware it is markedly sharper AND more
    # consistent than the 7B at the "everyday vs. obscure" call (7B left ~200
    # common words in a full run; 14B left 16, all genuinely borderline). If the
    # file is absent, get_judge() falls back to the stub; --stub forces it.
    model_path: str = "models/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    n_gpu_layers: int = -1        # -1 = offload everything the VRAM allows
    n_ctx: int = 8192
    judge_batch: int = 20

    # --- Enrichment (§03.9) ----------------------------------------------
    lookup_definitions: bool = True

    # --- Output (§03.10) -------------------------------------------------
    # No interactive review knobs: the shortlist is written whole and hand-edited,
    # then `concordance finalize` promotes the survivors.
    limit: int = 0                # 0 = no cap on the shortlist
