"""Master vocabulary list + per-book archiving (§07 follow-on).

When a book's shortlist has been fully reviewed, its approved terms are promoted
to a single cross-book ``master_vocab.csv`` at the project root — the durable,
export-friendly record a separate tracking app can later consume — and the
per-book files are moved into ``archive/`` so the working directory only ever
shows books still in flight.

Master policy (Brian's call): ONE row per word. If an already-promoted word turns
up again in a later book, we don't duplicate the row — we append the new book to
that word's ``source_book`` cell (so every book that surfaced it is recorded) and
keep the original ``date_added``.
"""

from __future__ import annotations

import csv
import os
import shutil
from datetime import date
from pathlib import Path

from .output import VOCAB_COLUMNS

MASTER_COLUMNS = VOCAB_COLUMNS + ["date_added", "source_book"]
MASTER_NAME = "master_vocab.csv"

# Per-book files to sweep into the archive, given the book stem. The source book
# itself (.epub/.pdf/.txt) is matched separately since its extension varies.
_ARTIFACT_SUFFIXES = (".vocab.csv", ".rejected.csv", ".enriched.csv", ".undefined.csv")
_BOOK_SUFFIXES = (".epub", ".pdf", ".txt")


def _sources(cell: str) -> list[str]:
    return [s.strip() for s in cell.split(";") if s.strip()]


def promote_to_master(
    approved: list[dict],
    book: str,
    master_path: Path,
    today: str | None = None,
) -> tuple[int, int]:
    """Merge approved per-book rows into the master CSV. Returns (added, merged):
    new words added, and existing words that gained ``book`` as a new source."""
    today = today or date.today().isoformat()

    order: list[str] = []
    existing: dict[str, dict] = {}
    if master_path.exists():
        with master_path.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                key = r["word"].strip().lower()
                existing[key] = r
                order.append(key)

    added = merged = 0
    for row in approved:
        key = row["word"].strip().lower()
        if key in existing:
            srcs = _sources(existing[key].get("source_book", ""))
            if book not in srcs:
                srcs.append(book)
                existing[key]["source_book"] = "; ".join(srcs)
                merged += 1
        else:
            new = {c: row.get(c, "") for c in VOCAB_COLUMNS}
            new["date_added"] = today
            new["source_book"] = book
            existing[key] = new
            order.append(key)
            added += 1

    _write_master(master_path, [existing[k] for k in order])
    return added, merged


def _write_master(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in MASTER_COLUMNS})
    os.replace(tmp, path)          # atomic; survives a drvfs/Excel lock on the target


def book_stem(vocab_path: Path) -> str:
    """'Shadow.vocab.csv' -> 'Shadow'. Falls back to plain stem for odd names."""
    name = vocab_path.name
    if name.endswith(".vocab.csv"):
        return name[: -len(".vocab.csv")]
    return vocab_path.with_suffix("").stem


def snapshot_original(vocab_path: Path, archive_dir: Path) -> Path:
    """Copy the freshly-generated candidate CSV into the archive as the pristine
    ``<stem>.vocab.original.csv``, BEFORE the user hand-edits the working copy —
    so both the original and the cleaned version survive. Overwrites a prior
    snapshot (a re-run of the same book supersedes it)."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / f"{book_stem(vocab_path)}.vocab.original.csv"
    shutil.copyfile(vocab_path, dest)
    return dest


def archive_book(vocab_path: Path, archive_dir: Path) -> tuple[list[Path], list[Path]]:
    """Move the per-book artifacts and the source book into ``archive_dir``.
    Returns (moved, failed). Never raises on a single locked/missing file."""
    stem = book_stem(vocab_path)
    base = vocab_path.parent
    archive_dir.mkdir(parents=True, exist_ok=True)

    candidates = [base / f"{stem}{suf}" for suf in _ARTIFACT_SUFFIXES]
    candidates += [base / f"{stem}{suf}" for suf in _BOOK_SUFFIXES]

    moved, failed = [], []
    for src in candidates:
        if not src.exists():
            continue
        try:
            shutil.move(str(src), str(archive_dir / src.name))
            moved.append(src)
        except (OSError, shutil.Error):
            failed.append(src)
    return moved, failed
