"""Per-volume orchestration: hash-check -> extract -> segment -> stopword
filter -> definition-entry filter -> parse -> pronunciation (composite,
double-pass) -> DB write.

Known gaps (documented rather than silently glossed over):
  * Entries are bounded to a single page. An entry whose body text continues
    onto the next page gets truncated — most OED entries are short enough
    that this is rare, but it's real and unhandled.
  * homograph_number (lead¹/lead²) and entry_type (run_on/compound
    detection) are not populated — see segment.py's docstring for why.
  * Sense/quotation parsing (parse.py) is a first-pass regex splitter, not
    calibrated against broad output yet.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import psycopg
from rich.console import Console

from . import db as oed_db
from . import extract, parse, pronunciation, segment, sequence
from .config import OedConfig


def _volume_label(path: Path) -> str:
    return path.stem[:120]


def ingest_volume(path: Path, conn: psycopg.Connection, cfg: OedConfig,
                   console: Console, *, force: bool = False,
                   transcriber: pronunciation.Transcriber | None = None,
                   page_limit: int | None = None) -> dict:
    """Ingest one PDF. Returns a small stats dict. Resumable: re-running on
    an already-'done' volume (same file_hash) is a no-op unless force=True;
    a volume that errored partway resumes from last_page_committed."""
    file_hash = oed_db.file_hash(path)
    existing = oed_db.get_volume(conn, file_hash, cfg.schema)
    if existing and existing["status"] == "done" and not force:
        console.print(f"[dim]{path.name}: already ingested (volume_id={existing['id']}), skipping[/dim]")
        return {"skipped": True}

    doc = fitz.open(path)
    page_count = len(doc)
    if existing:
        volume_id = existing["id"]
        start_page = existing["last_page_committed"]
    else:
        volume_id = oed_db.upsert_volume(
            conn, file_name=path.name, file_hash_=file_hash,
            volume_label=_volume_label(path), page_count=page_count, schema=cfg.schema,
        )
        start_page = 0

    oed_db.set_volume_status(conn, volume_id, "processing", schema=cfg.schema)
    if transcriber is None:
        transcriber = pronunciation.get_transcriber(cfg)

    end_page = min(page_count, start_page + page_limit) if page_limit else page_count
    entries_written = 0
    pending_crops: list[tuple[int, str, fitz.Page, list[float]]] = []

    try:
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            spans = extract.flatten_spans(page)
            if not spans:
                continue
            col_rows, margins = extract.page_columns(spans, page.rect.width)
            ordered = [row for rows in col_rows for row in rows]
            hits = segment.find_headwords_from_columns(col_rows, margins, page_num, cfg)
            if not hits:
                _commit_page(conn, volume_id, page_num, cfg)
                continue

            hit_row_idx = _match_rows_to_hits(ordered, hits)
            # hits is Y-order (top-to-bottom across both columns); `ordered`
            # is column-major (left column fully, then right). Slicing
            # ordered[start:end] between "consecutive" hits only makes sense
            # if the hits themselves are walked in ordered's own order — a
            # real bug found in the smoke test: a Y-order "next hit" from
            # the opposite column produced a boundary index that didn't
            # actually bound the same column, pulling unrelated entries'
            # text into raw_text. Re-sort by resolved row index (dropping
            # unmatched hits) before slicing.
            resolved = sorted(
                ((hit, idx) for hit, idx in zip(hits, hit_row_idx) if idx is not None),
                key=lambda pair: pair[1],
            )

            for i, (hit, start_idx) in enumerate(resolved):
                end_idx = resolved[i + 1][1] if i + 1 < len(resolved) else len(ordered)
                raw_text = " ".join(extract.row_text(r) for r in ordered[start_idx:end_idx])

                # Reject false-positive headword detections here, before any
                # DB write: an ordinary body-text word caught by the
                # size+margin heuristic (see segment.py) never has a
                # pronunciation bracket or POS abbreviation immediately
                # after it the way a real entry does. See
                # parse.looks_like_definition_entry's docstring.
                if not parse.looks_like_definition_entry(hit["text"], raw_text):
                    continue

                parsed = parse.parse_entry(hit["text"], raw_text)
                entry_id = oed_db.insert_entry(
                    conn, volume_id=volume_id, headword=hit["text"], homograph_number=None,
                    part_of_speech=parsed["part_of_speech"], etymology=parsed["etymology"],
                    entry_type="main", parent_entry_id=None, page_number=page_num,
                    raw_text=raw_text, schema=cfg.schema,
                )
                if parsed["senses"]:
                    oed_db.insert_definitions(conn, entry_id, parsed["senses"], schema=cfg.schema)
                entries_written += 1
                pending_crops.append((entry_id, hit["text"], page, hit["bbox"]))

            # `while`, not `if`: a single dense page can produce more than
            # one batch's worth of hits on its own (real pages measured
            # 15-34 hits each) — draining all full batches here rather than
            # carrying the excess forward keeps every composite capped at
            # composite_batch_size.
            while len(pending_crops) >= cfg.composite_batch_size:
                _flush_pronunciation(conn, pending_crops[:cfg.composite_batch_size], cfg, transcriber)
                pending_crops = pending_crops[cfg.composite_batch_size:]

            _commit_page(conn, volume_id, page_num, cfg)
            if entries_written and entries_written % 50 == 0:
                console.print(f"[dim]{path.name}: page {page_num}/{end_page}, {entries_written} entries[/dim]")

        # Trailing partial batch: chunked the same way, not passed whole —
        # an uncapped final flush is exactly what produced a real failure
        # (a leftover queue of 20+ crops built one oversized composite that
        # exceeded the model's context window: "Prompt exceeds n_ctx: 4103
        # > 4096"). Composites must never exceed composite_batch_size.
        for i in range(0, len(pending_crops), cfg.composite_batch_size):
            _flush_pronunciation(conn, pending_crops[i:i + cfg.composite_batch_size], cfg, transcriber)
        if pending_crops:
            conn.commit()

        final_status = "done" if end_page >= page_count else "pending"
        if final_status == "done":
            # Whole-volume ordering context only exists once every page has
            # been ingested, so this runs once here rather than per-page.
            # See sequence.py for why this is a real, separate signal from
            # looks_like_definition_entry (catches coincidental
            # bracket/POS matches on body-text noise, and OCR-garbled
            # headwords with a stray leading letter) -- confirmed with
            # Brian to prune both classes, not just flag them.
            cur = conn.cursor()
            cur.execute(
                f"select id, headword from {cfg.schema}.entry "
                f"where volume_id = %s order by id",
                (volume_id,),
            )
            rows = cur.fetchall()
            out_of_order = sequence.find_out_of_order_ids(rows)
            if out_of_order:
                cur.execute(
                    f"delete from {cfg.schema}.entry where id = any(%s)",
                    (list(out_of_order),),
                )
                console.print(f"[dim]{path.name}: pruned {len(out_of_order)} "
                               f"out-of-order entries[/dim]")
        oed_db.set_volume_status(conn, volume_id, final_status,
                                  last_page_committed=end_page, schema=cfg.schema)
    except Exception as exc:  # noqa: BLE001 — record and re-raise so the caller sees it
        oed_db.set_volume_status(conn, volume_id, "error", error_detail=str(exc)[:2000], schema=cfg.schema)
        conn.commit()
        raise

    return {"volume_id": volume_id, "entries_written": entries_written, "pages": end_page - start_page}


def _commit_page(conn: psycopg.Connection, volume_id: int, page_num: int, cfg: OedConfig) -> None:
    oed_db.set_volume_status(conn, volume_id, "processing",
                              last_page_committed=page_num + 1, schema=cfg.schema)


def _match_rows_to_hits(ordered: list[dict], hits: list[dict]) -> list[int | None]:
    """Index into `ordered` for each hit's row (matched by bbox), so entry
    text can be sliced from that index to the next hit's index. A single
    pass building a bbox->row-index lookup, rather than scanning every row
    per hit — matters once a page has thousands of spans."""
    lookup: dict[tuple[int, int], int] = {}
    for i, row in enumerate(ordered):
        for s in row["spans"]:
            key = (round(s["bbox"][0]), round(s["bbox"][1]))
            lookup.setdefault(key, i)
    result: list[int | None] = []
    for hit in hits:
        key = (round(hit["bbox"][0]), round(hit["bbox"][1]))
        result.append(lookup.get(key))
    return result


def _flush_pronunciation(conn: psycopg.Connection, batch: list[tuple[int, fitz.Page, list[float]]],
                          cfg: OedConfig, transcriber: pronunciation.Transcriber) -> None:
    """Render both passes for a batch of entries, composite each pass
    separately, transcribe, and write the resolved (or needs-review)
    pronunciation for every entry in the batch."""
    if not batch:
        return
    if isinstance(transcriber, pronunciation.StubTranscriber):
        # No model configured: skip the (real) cost of rendering crop
        # pixmaps and composites entirely — StubTranscriber.transcribe()
        # would just discard them. Still records raw_ocr for provenance.
        for entry_id, _headword, page, bbox in batch:
            rect = pronunciation.crop_rect(bbox, page.rect, cfg)
            raw_ocr = page.get_text("text", clip=rect).strip()
            oed_db.update_pronunciation(
                conn, entry_id, pronunciation_raw=raw_ocr, pass1=None, pass2=None,
                ipa=None, source=None, needs_review=True, schema=cfg.schema,
            )
        return

    pass1_items, pass2_items = [], []
    raw_ocr: dict[int, str] = {}
    for entry_id, headword, page, bbox in batch:
        rect = pronunciation.crop_rect(bbox, page.rect, cfg)
        pass1_items.append((entry_id, headword, pronunciation.render_crop(page, rect, cfg, zoom=cfg.render_zoom)))
        pass2_items.append((entry_id, headword, pronunciation.render_crop(page, rect, cfg, zoom=cfg.render_zoom * 1.3)))
        raw_ocr[entry_id] = page.get_text("text", clip=rect).strip()

    # Pass 2 uses a REVERSED item order and a different column count, so its
    # composite is a genuinely different layout from pass 1's — not just the
    # same grid re-rendered. Without this, a systematic cell/label
    # misalignment (a real failure mode found in testing — see
    # pronunciation.py's module docstring) would reproduce identically in
    # both passes and resolve_pronunciation would wrongly see "agreement".
    pass2_items = list(reversed(pass2_items))
    composite1, cells1 = pronunciation.build_composite(pass1_items, cfg, cols=cfg.composite_cols)
    composite2, cells2 = pronunciation.build_composite(pass2_items, cfg, cols=cfg.composite_cols + 1)
    result1 = transcriber.transcribe(composite1, cells1)
    result2 = transcriber.transcribe(composite2, cells2)

    for entry_id, _headword, _page, _bbox in batch:
        p1 = result1.get(entry_id)
        p2 = result2.get(entry_id)
        ipa, needs_review = pronunciation.resolve_pronunciation(p1, p2, raw_ocr.get(entry_id))
        oed_db.update_pronunciation(
            conn, entry_id, pronunciation_raw=raw_ocr.get(entry_id), pass1=p1, pass2=p2,
            ipa=ipa, source="vision_llm" if ipa else None, needs_review=needs_review, schema=cfg.schema,
        )
