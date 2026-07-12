"""Finalize a hand-edited candidate list (§03.10, revised workflow).

The review step is just editing the CSV: open ``<book>.vocab.csv``, delete the
rows for words you already know or that are false positives, save. Whatever rows
remain are what gets added — no status/approved column, no marking, just deletion.
Then::

    concordance finalize <book>.vocab.csv

adds those remaining terms to the master list (with date + book source) and moves
the per-book files into ``archive/``. The pristine, pre-edit copy was already
snapshotted to ``archive/<book>.vocab.original.csv`` when the list was generated,
so both the original and your cleaned version are preserved.

Reading is deliberately forgiving of what a spreadsheet does to a CSV: if Excel
has renamed the header row to ``Column1,Column2,…`` we fall back to reading by
column *position* (the fixed VOCAB_COLUMNS order), and any extra column you may
have added is ignored. So don't reorder columns if you've lost the header row.
"""

from __future__ import annotations

import csv
from pathlib import Path

from rich.console import Console

from . import master
from .output import VOCAB_COLUMNS


def _read_rows(path: Path) -> list[dict]:
    """Every non-empty row is a term to add. Tolerant of a spreadsheet-mangled
    header: map by name when the real header survives, else by fixed position."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        raw = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if not raw:
        return []

    header = [c.strip().lower() for c in raw[0]]
    if "word" in header:
        # Real header present (possibly with extra columns) — map by name.
        idx = {name: header.index(name) for name in VOCAB_COLUMNS if name in header}
        body = raw[1:]

        def cell(r, name):
            i = idx.get(name)
            return r[i] if i is not None and i < len(r) else ""
    else:
        # Header lost/renamed by a spreadsheet (or absent) — read positionally in
        # the canonical column order; skip a generic "Column1,…" header if present.
        generic = bool(header) and (header[0].startswith("column") or header[0] in ("", "1"))
        body = raw[1:] if generic else raw

        def cell(r, name):
            i = VOCAB_COLUMNS.index(name)
            return r[i] if i < len(r) else ""

    rows = []
    for r in body:
        row = {name: cell(r, name) for name in VOCAB_COLUMNS}
        if (row["word"] or "").strip():
            rows.append(row)
    return rows


def finalize_file(
    vocab_path: Path,
    console: Console | None = None,
    input_fn=None,
    master_path: Path | None = None,
    archive_dir: Path | None = None,
    assume_yes: bool = False,
) -> None:
    console = console or Console()
    input_fn = input_fn or console.input
    vocab_path = Path(vocab_path)
    master_path = master_path or vocab_path.parent / master.MASTER_NAME
    archive_dir = archive_dir or vocab_path.parent / "archive"
    stem = master.book_stem(vocab_path)

    rows = _read_rows(vocab_path)
    console.rule(f"[bold]Finalize {stem}[/bold]")
    console.print(
        f"[bold]{len(rows)}[/bold] term(s) remain in {vocab_path.name} "
        f"→ will be added to {master_path.name}, and the book's files archived."
    )
    if rows:
        preview = ", ".join(r["word"] for r in rows[:8])
        more = f", … (+{len(rows) - 8})" if len(rows) > 8 else ""
        console.print(f"[dim]{preview}{more}[/dim]")

    if not assume_yes:
        ans = input_fn("Proceed? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            console.print("[dim]Cancelled — nothing changed.[/dim]")
            return

    added, merged = master.promote_to_master(rows, book=stem, master_path=master_path)
    console.print(
        f"[green]✓[/green] master: [bold]{added}[/bold] new term(s), "
        f"{merged} existing word(s) gained {stem} as a source → {master_path.name}"
    )
    moved, failed = master.archive_book(vocab_path, archive_dir)
    if moved:
        console.print(f"[green]✓[/green] archived {len(moved)} file(s) → {archive_dir.name}/")
    for f in failed:
        console.print(f"[red]✗[/red] could not move {f.name} (open elsewhere?) — left in place.")
