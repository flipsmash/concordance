"""Tunable knobs for an OED-ingest run. Mirrors concordance/config.py's shape."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OedConfig:
    # --- Headword detection (segment.py) ----------------------------------
    # A headword is a flush-left row whose first word is printed bold and
    # which has extra leading above it. Measured on hand-labelled pages: true
    # entries have ink >= ~1.5 (stroke width / page median) and a gap >= ~1.4
    # (line pitches), other flush-left rows ink <= ~1.3 and gap <= ~1.24. Ink
    # above headword_ink_max is a smudge or a mis-boxed span, not type.
    headword_ink_min: float = 1.4
    headword_ink_max: float = 3.0
    headword_gap_min: float = 1.2
    ink_zoom: float = 4.0             # render scale for measuring stroke width
    # Headword candidates must start within this many points of the column's
    # left text margin: senses and quotations are indented, entries aren't.
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
