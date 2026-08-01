"""Pronunciation extraction — crop, composite-batch, double-pass transcribe.

Grounded in a real pilot (see project notes / conversation history), not
assumption:
  * The baked-in OCR text layer is NEVER used for pronunciation. It's
    confirmed to silently mis-map IPA glyphs onto plausible-looking-but-wrong
    ASCII (e.g. real "(ˈækʃənəblɪ)" came back as "('aekjanabli)" — every
    character individually plausible, the transcription wrong). It's kept
    only as `pronunciation_raw` provenance, never surfaced as truth.
  * Precise span-level bbox pairing (headword -> its pronunciation bracket)
    is unreliable (~35% clean pairing even after tuning — see segment.py) —
    so the crop is a generous, fixed-size region anchored on the headword's
    y-position, and the vision model locates the bracket itself.
  * A single vision-LLM pass is NOT a sufficient gate: in-set IPA
    substitutions (ɪ->i, ə->ʌ, ʃ->ʒ) are exactly the silent-wrong-answer
    failure mode this whole module exists to prevent, and an allowlist alone
    can't catch them (they're all valid IPA characters). Two independent
    reads (different crop render, same model) + a character-level agreement
    check is the real gate — pilot measured 26/28 (92.9%) clean agreement,
    with disagreements landing exactly on genuinely ambiguous glyphs.
  * Composite batching (multiple crops in one image) is pilot-validated: 8
    entries per composite, downscaled to ~1568px wide, stayed legible to two
    independent readers. This is what makes double-pass affordable at
    OED-volume scale.
  * The cell-number label is NOT a trustworthy alignment key on its own —
    caught in real testing: the model read the grid in its own column-major
    order and numbered panels 1..8 by that order rather than reading the
    small "#N" labels (which turned out illegible after the composite's
    downscale — default PIL bitmap font shrunk to a few px). Every individual
    transcription was correct; they were just filed under the wrong cell
    number. Two fixes: (1) labels are drawn AFTER the downscale, at a fixed
    legible pixel size, so they survive; (2) the model is asked to echo the
    headword it read alongside each pronunciation, and that echo is matched
    against the real headword before a transcription is trusted at all — so
    alignment is verified, not assumed. Critically, (1) alone would NOT have
    been sufficient with the double-pass gate as originally built: pass 1
    and pass 2 used the same composite layout, so a systematic
    mis-numbering would reproduce identically in both passes and
    resolve_pronunciation would see "agreement" on consistently mismatched
    data — the exact silent-wrong-answer failure this module exists to
    prevent. Pass 2 therefore uses a different cell ordering (batch order
    reversed, different column count) so the two passes aren't just reading
    the same layout twice.

  * Found running a real (later aborted) production ingest: ~20% of entries
    that CLEARED the double-pass gate were missing a real, clearly-visible
    leading unstressed vowel (e.g. "ablaze" -> "ˈbleɪz" instead of the
    correct "əˈbleɪz" — confirmed against both the crop image, read by eye,
    and pronunciation_raw's baked-OCR text, which had it: "(a'bleiz)"). Both
    independent passes dropped the SAME character the same way — a shared
    model bias, not per-pass noise, so double-pass agreement didn't catch it
    (the "sporadic" failure mode from the original human pilot: agreement
    only catches independent errors, not a bias both passes share).
    ROOT CAUSE, found by isolating variables: not a legibility problem (the
    schwa was plainly readable by eye at every zoom tested) and not fixed
    reliably by prompting alone (an explicit worked-example warning in
    _PROMPT fixed 5/6 known-affected words, but the 6th kept failing even
    when it wasn't the example used). What DID fix it 3/3: cropping to just
    the headword+pronunciation line, excluding the definition/quotation
    text below. The extra visual context below the bracket was the actual
    trigger — see config.py's crop_height_pt (34.0 -> 16.0) for the real
    fix and how it was validated (spot-checked across both volumes for
    bracket truncation before adopting).
    The prompt warning and the raw-OCR cross-check (resolve_pronunciation,
    below) are kept as defense-in-depth rather than removed now that the
    root cause has a fix — confirmed live and independently useful: on a
    fresh 9-page real batch (pre-height-fix, prompt-fix only), the cross-
    check caught 3 real cases where both passes agreed on the same wrong
    (vowel-dropped) answer and blocked them from resolving. A different
    trigger for the same underlying model bias could still exist and slip
    past the height fix; this isn't a claim the failure mode is now
    provably eliminated. Worth an occasional audit (spot-check a random
    sample of resolved pronunciation_ipa values against pronunciation_raw)
    rather than treating a cleared needs_review as a permanently closed
    question.

Nothing here claims a pronunciation is correct unless both passes agree
character-for-character AND both echoed the right headword; everything else
is written with pronunciation_needs_review=True and pronunciation_ipa=NULL.
"""

from __future__ import annotations

import base64
import difflib
import io
import re
from typing import Protocol

import fitz
from PIL import Image, ImageDraw, ImageFont

from .config import OedConfig

# The pronunciation symbol set actually observed in OED2 pilot crops (stress/
# length marks, common IPA vowels/consonants, syllabic parens, hyphens for
# variant/compound forms). A transcription containing anything outside this
# set is rejected outright regardless of pass agreement — cheap insurance
# against gross misreads, though pass-agreement (above) is the real gate.
_ALLOWED_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "æœðŋʃʒθʔɑɒɔəɛɜɡɪɨɵʊʌʍθɚɝɹɾ"
    "ˈˌːˑ.,-'()/ "
)


def crop_rect(headword_bbox: list[float], page_rect: fitz.Rect, cfg: OedConfig) -> fitz.Rect:
    x0, y0, x1, y1 = headword_bbox
    return fitz.Rect(
        max(0, x0 - 6),
        max(0, y0 - cfg.crop_top_pad_pt),
        min(page_rect.width, x0 + cfg.crop_width_pt),
        min(page_rect.height, y1 + cfg.crop_height_pt),
    )


def render_crop(page: fitz.Page, rect: fitz.Rect, cfg: OedConfig, *, zoom: float | None = None) -> Image.Image:
    mat = fitz.Matrix(zoom or cfg.render_zoom, zoom or cfg.render_zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def build_composite(items: list[tuple[int, str, Image.Image]], cfg: OedConfig,
                     *, cols: int | None = None) -> tuple[Image.Image, dict[int, tuple[int, str]]]:
    """items: [(entry_id, headword, crop_image)]. Returns (composite_image,
    {cell_number: (entry_id, headword)}) — the headword is what
    LocalVisionTranscriber verifies the model's echo against; cell_number
    alone is not trusted as an alignment key (see module docstring)."""
    cols = cols or cfg.composite_cols
    cell_w = max(im.width for _, _, im in items)
    cell_h = max(im.height for _, _, im in items) + 20
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cell_w * cols + 10 * (cols + 1), cell_h * rows + 10 * (rows + 1)), "white")
    cell_meta: dict[int, tuple[int, str]] = {}
    positions: list[tuple[int, int]] = []
    for i, (entry_id, headword, im) in enumerate(items):
        r, c = divmod(i, cols)
        x = 10 + c * (cell_w + 10)
        y = 10 + r * (cell_h + 10)
        canvas.paste(im, (x, y + 18))
        cell_meta[i + 1] = (entry_id, headword)
        positions.append((x, y))

    # Downscale FIRST, then draw labels — drawing them on the pre-resize
    # canvas at a small font meant they shrank below legibility (confirmed:
    # PIL's default bitmap font at ~11px survived a 3x downscale as ~3px,
    # and the model silently fell back to its own left-to-right/top-to-
    # bottom reading order instead of using them).
    scale = 1.0
    if canvas.width > cfg.composite_max_width_px:
        scale = cfg.composite_max_width_px / canvas.width
        canvas = canvas.resize((cfg.composite_max_width_px, int(canvas.height * scale)))

    draw = ImageDraw.Draw(canvas)
    font_size = max(16, int(cell_h * scale * 0.22))
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # older Pillow without the size= param
        font = ImageFont.load_default()
    for i, (x, y) in enumerate(positions):
        lx, ly = x * scale, y * scale
        label = f"#{i + 1}"
        bbox = draw.textbbox((lx, ly), label, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill="yellow")
        draw.text((lx, ly), label, fill="red", font=font)
    return canvas, cell_meta


def _to_data_uri(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def valid_ipa(text: str) -> bool:
    return bool(text) and all(ch.lower() in _ALLOWED_CHARS for ch in text)


class Transcriber(Protocol):
    def transcribe(self, composite: Image.Image, cell_meta: dict[int, tuple[int, str]]) -> dict[int, str]:
        """cell_meta: {cell_number: (entry_id, expected_headword)}. Returns
        {entry_id: pronunciation_text} — only for cells where the model's
        echoed headword actually matched the expected one. Missing keys
        mean no usable (or unverifiable) answer for that entry."""
        ...


class StubTranscriber:
    """No vision model available: every entry stays needs_review=True with
    no pronunciation written. Lets the rest of the pipeline (headword/POS/
    etymology/definitions/quotations) run end-to-end without one — same
    fallback pattern as judge.py's StubJudge."""

    def transcribe(self, composite: Image.Image, cell_meta: dict[int, tuple[int, str]]) -> dict[int, str]:
        return {}


def _normalize_headword(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def _headword_matches(expected: str, echoed: str) -> bool:
    """Fuzzy match — the model's echo is read from the same crop as the
    pronunciation, via the same imperfect vision process, so it won't always
    be byte-identical to segment.py's OCR-derived headword even when the
    model looked at the right panel. This is only an alignment check, not a
    pronunciation-accuracy check — a loose threshold is fine here."""
    a, b = _normalize_headword(expected), _normalize_headword(echoed)
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.7


_PROMPT = (
    "This image shows numbered cells (each marked with a yellow-highlighted "
    "#N label) cropped from a scanned Oxford English Dictionary page. Each "
    "cell has a bold headword followed by a pronunciation in parentheses "
    "using IPA-like phonetic notation (stress marks ˈ ˌ, schwa ə, etc.), "
    "then the start of a definition.\n\n"
    "For each numbered cell: read the #N label itself (don't infer the "
    "number from position), read the bold headword, and read the ENTIRE "
    "text inside the parentheses immediately after it, from the very first "
    "character to the very last, transcribing every phonetic symbol as "
    "precisely as you can.\n\n"
    "IMPORTANT — a common mistake is dropping a leading unstressed vowel "
    "(schwa ə, or a) that appears BEFORE the first stress mark ˈ. For "
    'example the pronunciation "(a\'bAndAn)" must be transcribed in full as '
    '"əˈbændən" — NOT as "ˈbændən" (which wrongly drops the leading ə). '
    "Always check whether there is a character before the first ˈ or ˌ "
    "stress mark and include it, even if it looks like a small or faint "
    "mark right next to the opening parenthesis.\n\n"
    "If a cell's pronunciation is cut off, blurry, or you are not "
    "confident, omit that cell entirely rather than guessing.\n\n"
    'Output ONLY a JSON array, one object per cell: '
    '[{"cell": 1, "headword": "actionably", "pronunciation": "ˈækʃənəblɪ"}, '
    '{"cell": 2, "headword": "abandon", "pronunciation": "əˈbændən"}, '
    '...]. No prose, no code fences.'
)


class LocalVisionTranscriber:
    """Qwen2.5-VL via llama-cpp-python. Falls back to StubTranscriber's
    behavior (via get_transcriber below) if the model/mmproj files aren't
    present — pronunciation extraction is optional infra, not a hard
    dependency, same as the text judge."""

    def __init__(self, cfg: OedConfig):
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler

        self.cfg = cfg
        handler = Qwen25VLChatHandler(clip_model_path=cfg.vision_mmproj_path, verbose=False)
        self.llm = Llama(
            model_path=cfg.vision_model_path,
            chat_handler=handler,
            n_gpu_layers=cfg.n_gpu_layers,
            n_ctx=cfg.n_ctx,
            verbose=False,
        )

    def transcribe(self, composite: Image.Image, cell_meta: dict[int, tuple[int, str]]) -> dict[int, str]:
        import json

        data_uri = _to_data_uri(composite)
        out = self.llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": _PROMPT},
                ],
            }],
            temperature=0.0,
            max_tokens=80 * len(cell_meta) + 128,
        )
        text = out["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("\n") + 1:] if "\n" in text else text
        try:
            rows = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(rows, list):
            return {}

        result: dict[int, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            cell = row.get("cell")
            pron = row.get("pronunciation")
            echoed = row.get("headword")
            if not isinstance(cell, int) or not isinstance(pron, str) or not isinstance(echoed, str):
                continue
            meta = cell_meta.get(cell)
            if meta is None:
                continue
            entry_id, expected_headword = meta
            if not _headword_matches(expected_headword, echoed):
                continue  # alignment unverifiable — don't guess which entry this belongs to
            result[entry_id] = pron
        return result


def get_transcriber(cfg: OedConfig) -> Transcriber:
    from pathlib import Path

    if Path(cfg.vision_model_path).exists() and Path(cfg.vision_mmproj_path).exists():
        try:
            return LocalVisionTranscriber(cfg)
        except Exception as exc:  # noqa: BLE001 — fall back rather than crash a run
            print(f"[pronunciation] could not load vision model ({exc}); using stub.")
    return StubTranscriber()


_RAW_BRACKET_RE = re.compile(r"\(([^)]{1,40})\)")
_STRESS_MARK_RE = re.compile(r"[ˈˌ]")


def _raw_suggests_leading_char(raw_ocr: str | None) -> bool | None:
    """From the baked-OCR text (unreliable for exact glyphs, but reliable
    enough to answer a structural yes/no): does the pronunciation bracket
    have a character before its first stress mark ('\'' is how the baked
    OCR consistently renders ˈ/ˌ — confirmed across every raw_ocr sample
    checked)? Returns None if no bracket was found to check at all, so
    resolve_pronunciation can skip the cross-check rather than wrongly
    penalize a case it can't evaluate."""
    if not raw_ocr:
        return None
    m = _RAW_BRACKET_RE.search(raw_ocr)
    if not m:
        return None
    bracket = m.group(1)
    idx = bracket.find("'")
    if idx == -1:
        return None
    return idx > 0


def _ipa_has_leading_char(ipa: str) -> bool | None:
    m = _STRESS_MARK_RE.search(ipa)
    if not m:
        return None
    return m.start() > 0


def resolve_pronunciation(pass1: str | None, pass2: str | None,
                           raw_ocr: str | None = None) -> tuple[str | None, bool]:
    """(ipa_or_None, needs_review). Agreement (after whitespace/paren
    normalization) on two valid-IPA-charset reads is the primary gate —
    matches the pilot's double-pass design, not a single-pass allowlist
    check. On top of that: a real, measured failure mode (see module
    docstring) is BOTH passes sharing the same bias — dropping a real,
    visible leading unstressed vowel before the first stress mark — which
    pass-agreement alone can't catch since it's not independent noise.
    raw_ocr (unreliable for exact glyphs, but structurally informative) is
    cross-checked for this one specific pattern: if it clearly shows a
    leading character before the stress mark and the agreed transcription
    doesn't, that overrides an otherwise-clean agreement back to
    needs_review=True rather than trusting a bias both passes share."""
    def norm(s: str | None) -> str | None:
        if s is None:
            return None
        return s.strip().strip("()").strip()

    n1, n2 = norm(pass1), norm(pass2)
    if not (n1 and n2 and n1 == n2 and valid_ipa(n1)):
        return None, True

    raw_says = _raw_suggests_leading_char(raw_ocr)
    ipa_says = _ipa_has_leading_char(n1)
    if raw_says is True and ipa_says is False:
        return None, True
    return n1, False
