"""Per-volume orchestration. `segment_page` is the one place a page turns into
entries; `ingest_volume` writes them for a new volume (plus pronunciation), and
`rescan_volume` reconciles them with what's already stored.

Known gaps (documented rather than silently glossed over):
  * Entries are bounded to a single page. An entry whose body text continues
    onto the next page gets truncated.
  * homograph_number (lead¹/lead²) and run-on sub-entries are not populated --
    see segment.py's docstring for why.
  * Sense/quotation parsing (parse.py) is a first-pass regex splitter, not
    calibrated against broad output yet.
"""

from __future__ import annotations

import collections
from pathlib import Path

import fitz
import psycopg
from rich.console import Console

from . import db as oed_db
from . import extract, parse, pronunciation, segment, sequence
from .config import OedConfig

_RESCAN_COMMIT_EVERY = 25           # pages


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
            _ordered, entries = segment_page(page, cfg)
            for e in entries:
                for headword in (e["headword"], *(e["also"] if e["pointer"] else ())):
                    entry_id = _insert_entry(conn, volume_id, page_num, e, headword, cfg)
                    entries_written += 1
                    if not e["pointer"]:    # a pointer has no pronunciation of its own
                        pending_crops.append((entry_id, headword, page, e["bbox"]))

            # `while`, not `if`: a single dense page can produce more than
            # one batch's worth of hits on its own (real pages measured
            # 15-34 hits each) — draining all full batches here rather than
            # carrying the excess forward keeps every composite capped at
            # composite_batch_size.
            while len(pending_crops) >= cfg.composite_batch_size:
                _flush_pronunciation(conn, pending_crops[:cfg.composite_batch_size], cfg, transcriber)
                pending_crops = pending_crops[cfg.composite_batch_size:]

            _commit_page(conn, volume_id, page_num, cfg)
            if page_num % 50 == 0:
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
            pruned = _prune_out_of_order(conn, volume_id, cfg)
            if pruned:
                console.print(f"[dim]{path.name}: pruned {pruned} out-of-order entries[/dim]")
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


def segment_page(page: fitz.Page, cfg: OedConfig) -> tuple[list[dict], list[dict]]:
    """A page's rows in reading order and its entries: segment.find_headwords'
    hits, in that same order, each with `row_index`, `raw_text` (from its row
    up to the next entry's), and `pointer`/`pointer_target`."""
    spans = extract.flatten_spans(page)
    if not spans:
        return [], []
    col_rows, margins = extract.page_columns(spans, page.rect.width)
    gray = extract.page_gray(page, cfg.ink_zoom)
    hits = segment.find_headwords(col_rows, margins, page.number, cfg, page.rect.height, gray)
    top = segment.body_top([s for rows in col_rows for r in rows for s in r["spans"]], page.rect.height)
    ordered = [row for rows in col_rows for row in rows if extract.row_top(row) > top]
    index = {id(row): i for i, row in enumerate(ordered)}
    hits.sort(key=lambda h: index[h["row_id"]])
    entries = []
    for i, hit in enumerate(hits):
        start = index[hit["row_id"]]
        end = index[hits[i + 1]["row_id"]] if i + 1 < len(hits) else len(ordered)
        raw_text = " ".join(extract.row_text(r) for r in ordered[start:end])
        rest = raw_text[len(hit["text"]):] if raw_text.startswith(hit["text"]) else raw_text
        pointer, target = segment.classify_pointer(rest)
        entries.append({**hit, "row_index": start, "raw_text": raw_text, "pointer": pointer,
                        "pointer_target": target})
    return ordered, entries


def _insert_entry(conn: psycopg.Connection, volume_id: int, page_num: int, e: dict, headword: str,
                  cfg: OedConfig) -> int:
    parsed = _parse(e)
    entry_id = oed_db.insert_entry(
        conn, volume_id=volume_id, headword=headword, homograph_number=None,
        part_of_speech=parsed["part_of_speech"], etymology=parsed["etymology"],
        entry_type="pointer" if e["pointer"] else "main", parent_entry_id=None, page_number=page_num,
        raw_text=e["raw_text"], schema=cfg.schema, is_obsolete=e["obsolete"], detector="v3",
        headword_bbox=e["bbox"], pointer_target=e["pointer_target"],
    )
    if parsed["senses"]:
        oed_db.insert_definitions(conn, entry_id, parsed["senses"], schema=cfg.schema)
    return entry_id


def _parse(e: dict) -> dict:
    parsed = parse.parse_entry(e["text"], e["raw_text"])
    if e["pointer"]:
        parsed["senses"] = []           # the pointer's text is not a definition of the word
    return parsed


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


def _prune_out_of_order(conn: psycopg.Connection, volume_id: int, cfg: OedConfig,
                        only_detector: str | None = None) -> int:
    """sequence.py's whole-volume alphabetical-order cleanup (see
    ingest_volume) -- deletes entries out of first-letter order.

    Ordered by page, then first letter within a page, NOT by id: a rescan
    APPENDS new rows after the old ones, and an id-ordered check read that as
    massively out of order -- on 2026-09-24 it deleted 14,244 original entries
    (and ~6k new ones) across the multi-letter volumes before this was caught;
    restored from the 04:00 backup. Id order within a page is meaningless too,
    which matters on the page where one letter ends and the next begins."""
    with conn.cursor() as cur:
        cur.execute(f"select id, headword, detector, page_number from {cfg.schema}.entry where volume_id = %s",
                    (volume_id,))
        rows = cur.fetchall()
        rows.sort(key=lambda r: (r[3], sequence._first_letter(r[1]), r[0]))
        out_of_order = sequence.find_out_of_order_ids([(r[0], r[1]) for r in rows])
        if only_detector:
            # The check sees the whole volume but may only delete this pass's
            # own rows: LNDS ties are broken arbitrarily, and a noisy neighbour
            # ("fthro") once cost a real original entry ("throat").
            det = {r[0]: r[2] for r in rows}
            out_of_order = {i for i in out_of_order if det[i] == only_detector}
        if out_of_order:
            cur.execute(f"delete from {cfg.schema}.entry where id = any(%s)", (list(out_of_order),))
    return len(out_of_order)


def _locate(raw_text: str, row_texts: list[str]) -> int | None:
    """The row a stored entry begins on: its text started with that row's."""
    best = None
    for i, text in enumerate(row_texts):
        if raw_text.startswith(text) and (best is None or len(text) > len(row_texts[best])):
            best = i
    return best


def _update_entry(conn: psycopg.Connection, st: dict, e: dict, headword: str, cfg: OedConfig, stats: dict) -> None:
    """Bring one stored entry in line with its re-detected page position,
    leaving its id and pronunciation work alone. Its definitions are rebuilt
    only if its text or type changed; a changed headword or part of speech
    sends it back through the lemma and concordance checks."""
    s = cfg.schema
    entry_type = "pointer" if e["pointer"] else st["entry_type"]
    parsed = _parse(e)
    text_changed = st["raw_text"] != e["raw_text"] or st["entry_type"] != entry_type
    renamed = st["headword"] != headword
    recheck = renamed or st["part_of_speech"] != parsed["part_of_speech"]
    stats["resliced"] += st["raw_text"] != e["raw_text"]
    stats["renamed"] += renamed
    stats["became_pointers"] += entry_type == "pointer" and st["entry_type"] != "pointer"
    with conn.cursor() as cur:
        cur.execute(
            f"""UPDATE {s}.entry
                SET headword = %s, raw_text = %s, part_of_speech = %s, etymology = %s, entry_type = %s,
                    pointer_target = %s, is_obsolete = %s, headword_bbox = %s,
                    lemma_computed_at = CASE WHEN %s THEN NULL ELSE lemma_computed_at END,
                    concordance_match_checked_at = CASE WHEN %s THEN NULL ELSE concordance_match_checked_at END
                WHERE id = %s""",
            (headword, e["raw_text"], parsed["part_of_speech"], parsed["etymology"], entry_type,
             e["pointer_target"], e["obsolete"], e["bbox"], recheck, recheck, st["id"]))
        if text_changed:
            cur.execute(f"DELETE FROM {s}.definition WHERE entry_id = %s", (st["id"],))
    if text_changed and parsed["senses"]:
        oed_db.insert_definitions(conn, st["id"], parsed["senses"], schema=s)


def new_rescan_stats(volume_id: int | None = None, pages: int = 0) -> dict:
    return {"volume_id": volume_id, "pages": pages, "pruned": 0, "duplicates": [], "unconfirmed": [], "unplaced": [],
            **dict.fromkeys(("detected", "pointers", "matched", "added", "added_pointers",
                             "resliced", "renamed", "became_pointers"), 0)}


def reconcile_page(conn: psycopg.Connection | None, volume_id: int, page_num: int, stored: list[dict],
                   ordered: list[dict], entries: list[dict], cfg: OedConfig, stats: dict) -> None:
    """Match one page's detected entries to its stored rows, updating matches
    in place and inserting the rest. A stored row belongs to the entry that
    starts on the row its text starts on -- spelling can't decide that, since
    fixing the spelling is part of the job ("fsleng" -> "sleng") -- with the
    headword key as the fallback for a row whose text no longer locates. Stored
    rows nothing matches are reported, not touched. conn=None is a dry run."""
    row_texts = [extract.row_text(r) for r in ordered]
    on_row: dict[int, list[dict]] = collections.defaultdict(list)
    unplaced: list[dict] = []
    for st in sorted(stored, key=lambda st: st["pronunciation_pass1"] is None):   # keep transcribed rows first
        idx = _locate(st["raw_text"], row_texts)
        (unplaced if idx is None else on_row[idx]).append(st)

    def take(candidates: list[dict], key: str, any_key: bool = False) -> dict | None:
        for st in candidates:
            if any_key or segment.headword_key(st["headword"]) == key:
                candidates.remove(st)
                return st
        return None

    for e in entries:
        here = on_row[e["row_index"]]
        for i, headword in enumerate((e["headword"], *(e["also"] if e["pointer"] else ()))):
            stats["detected"] += 1
            stats["pointers"] += e["pointer"]
            key = segment._key(headword)
            st = take(here, key) or take(unplaced, key) or (take(here, key, any_key=True) if i == 0 else None)
            if st:
                stats["matched"] += 1
                if conn:
                    _update_entry(conn, st, e, headword, cfg, stats)
            else:
                stats["added"] += 1
                stats["added_pointers"] += e["pointer"]
                if conn:
                    _insert_entry(conn, volume_id, page_num, e, headword, cfg)

    claimed_rows = {e["row_index"] for e in entries}
    for idx, rest in on_row.items():
        for st in rest:
            kind = "duplicates" if idx in claimed_rows else "unconfirmed"
            stats[kind].append({"id": st["id"], "page": page_num, "headword": st["headword"],
                                "row": row_texts[idx][:70]})
    stats["unplaced"].extend({"id": st["id"], "page": page_num, "headword": st["headword"]} for st in unplaced)


def rescan_volume(path: Path, conn: psycopg.Connection, cfg: OedConfig, *, page_limit: int | None = None,
                  dry_run: bool = False) -> dict:
    """`oed-ingest --rescan`: re-detect an ingested volume and reconcile it with
    what's stored -- new entries inserted, existing ones re-sliced in place so
    none keeps text that belongs to the entries after it, nothing re-transcribed.
    Idempotent. New entries are left for `oed-pronounce`, the GPU pass."""
    existing = oed_db.get_volume(conn, oed_db.file_hash(path), cfg.schema)
    if not existing:
        raise RuntimeError(f"{path.name}: not ingested yet -- run plain `oed-ingest` first")
    volume_id = existing["id"]
    doc = fitz.open(path)
    end_page = min(len(doc), page_limit) if page_limit else len(doc)
    stats = new_rescan_stats(volume_id, end_page)

    with conn.cursor() as cur:
        cur.execute(f"""select id, page_number, headword, raw_text, entry_type, part_of_speech, pronunciation_pass1
                        from {cfg.schema}.entry where volume_id = %s""", (volume_id,))
        stored: dict[int, list[dict]] = collections.defaultdict(list)
        for row in cur.fetchall():
            stored[row[1]].append({"id": row[0], "headword": row[2], "raw_text": row[3], "entry_type": row[4],
                                   "part_of_speech": row[5], "pronunciation_pass1": row[6]})

    for page_num in range(end_page):
        ordered, entries = segment_page(doc[page_num], cfg)
        if entries:
            reconcile_page(None if dry_run else conn, volume_id, page_num, stored.get(page_num, []),
                           ordered, entries, cfg, stats)
        else:
            stats["unplaced"].extend({"id": st["id"], "page": page_num, "headword": st["headword"]}
                                     for st in stored.get(page_num, []))
        if not dry_run and page_num % _RESCAN_COMMIT_EVERY == 0:
            conn.commit()
    if not dry_run:
        stats["pruned"] = _prune_out_of_order(conn, volume_id, cfg, only_detector="v3")
        conn.commit()
    return stats


def rescan_job(path: Path, database_url: str, cfg: OedConfig, page_limit: int | None, dry_run: bool) -> dict:
    """rescan_volume in a worker process, on its own connection."""
    with psycopg.connect(database_url) as conn:
        return rescan_volume(path, conn, cfg, page_limit=page_limit, dry_run=dry_run)


def pronounce_missing(path: Path, conn: psycopg.Connection, cfg: OedConfig, console: Console,
                      transcriber: pronunciation.Transcriber, *, limit: int = 0) -> dict:
    """`oed-pronounce`: the GPU half of `--rescan` -- run the usual double-pass
    pronunciation (_flush_pronunciation) for this volume's entries that were
    never attempted and carry a stored headword_bbox. Pointers have no
    pronunciation to read. Resumable: each written batch stops being "never
    attempted", so an interrupted run picks up where it left off."""
    existing = oed_db.get_volume(conn, oed_db.file_hash(path), cfg.schema)
    if not existing:
        raise RuntimeError(f"{path.name}: not ingested")
    with conn.cursor() as cur:
        cur.execute(
            f"""select id, headword, page_number, headword_bbox from {cfg.schema}.entry
                where volume_id = %s and headword_bbox is not null and entry_type <> 'pointer'
                  and pronunciation_raw is null and pronunciation_pass1 is null and pronunciation_pass2 is null
                order by page_number, id""" + (f" limit {int(limit)}" if limit else ""),
            (existing["id"],))
        rows = cur.fetchall()
    doc = fitz.open(path)
    batch: list = []
    done = 0
    for entry_id, headword, page_num, bbox in rows:
        batch.append((entry_id, headword, doc[page_num], list(bbox)))
        if len(batch) >= cfg.composite_batch_size:
            _flush_pronunciation(conn, batch, cfg, transcriber)
            conn.commit()
            done += len(batch)
            batch = []
            if done % 200 == 0:
                console.print(f"[dim]{path.name}: {done}/{len(rows)} pronounced[/dim]")
    if batch:
        _flush_pronunciation(conn, batch, cfg, transcriber)
        conn.commit()
        done += len(batch)
    return {"pronounced": done, "pending": len(rows)}
