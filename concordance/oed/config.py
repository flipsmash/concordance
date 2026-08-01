"""Tunable knobs for an OED-ingest run. Mirrors concordance/config.py's shape."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OedConfig:
    # --- Headword detection (segment.py) ----------------------------------
    # A line-start span at least this many times the page's body-text size
    # is treated as a headword candidate. This has a real, measured
    # precision ceiling on the baked-OCR font-size estimates: on one real
    # page, the true headword "abbreviate" (7.05) and the false-positive
    # body word "compendium." (numerically indistinguishable in the same
    # size band) could NOT be separated by any threshold value — some real
    # headwords and some OCR-noise-inflated body words occupy the same size
    # range. 1.30 was chosen as a measured tradeoff (cuts a real page's
    # false-positive hits from 133->97 while losing only a couple of
    # legitimate small-size headwords), not a value that eliminates the
    # problem — it doesn't fully exist to eliminate. See segment.py.
    headword_size_mult: float = 1.30
    # Headword candidates must start within this many points of the page's
    # left text margin (per column) — kills the dominant false-positive class
    # found in the pilot: small-caps cross-references inside body text
    # (e.g. "ABBOT" inside "abbotship"'s etymology) get flagged as
    # headword-sized but never start flush at the column margin the way a
    # real entry does.
    left_margin_tolerance: float = 4.0

    # --- Pronunciation crop (pronunciation.py) -----------------------------
    # Fixed-size crop anchored on the headword's y-position rather than a
    # precisely paired pronunciation-span bbox — the pilot found span-level
    # pairing via the baked OCR layer's line-grouping unreliable (~35% clean
    # pairing even after tuning), so the crop is deliberately wide and the
    # transcriber locates the bracket itself.
    crop_width_pt: float = 260.0
    # Root-caused a real bug at this default: extra definition/quotation
    # text below the headword's own line (visible when this was 34.0) made
    # the vision model consistently drop a real, visible leading unstressed
    # vowel before the first stress mark (e.g. "ablaze" -> "ˈbleɪz" instead
    # of "əˈbleɪz") — confirmed by isolating just the headword+pronunciation
    # line, which fixed it on every affected word tested (3/3). 16.0 is
    # tall enough for the OCR's own row height plus slack for the y-jitter
    # between a headword's row and its pronunciation bracket's row (still
    # occasionally grouped as separate row objects despite being on the
    # same physical line — see extract.group_rows), but tight enough to
    # exclude the next line's body text. Spot-checked across both volumes
    # after this change: no bracket truncation on any real entry.
    crop_height_pt: float = 16.0
    crop_top_pad_pt: float = 3.0
    render_zoom: float = 9.0          # matrix scale used for the crop render
    composite_cols: int = 2
    composite_batch_size: int = 8     # crops per composite image (pilot-validated)
    composite_max_width_px: int = 1568

    # --- Local vision model (pronunciation.py) -----------------------------
    # Falls back to StubTranscriber (needs_review=True, no pronunciation
    # written) if either file is absent — same "model optional" pattern as
    # judge.py's LlamaJudge/StubJudge. Qwen2.5-VL chosen for consistency
    # with the text judge's existing Qwen family and llama-cpp-python's
    # built-in Qwen25VLChatHandler support.
    vision_model_path: str = "models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    vision_mmproj_path: str = "models/Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"
    n_gpu_layers: int = -1
    n_ctx: int = 4096

    # --- Volume processing (pipeline.py) -----------------------------------
    # Commit after this many entries so a crash partway through a
    # 1000+ page volume doesn't lose the whole run.
    commit_every: int = 200
    schema: str = "oed"
