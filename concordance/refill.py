"""Backfill definitions into an existing ``<book>.vocab.csv`` (§03.9).

The judge pass is the expensive part of a run (~15 min on the 7B). When an early
run wrote its shortlist but the definition lookups came back empty — e.g. the
enrichment host throttled hundreds of rapid requests — there's no need to redo
the judging. This reruns *only* the enrichment stage over the CSV in place,
touching just the rows whose ``definition`` is still blank.

    python -m concordance.refill <book>.vocab.csv

Idempotent: rows that already have a definition are left alone, so it is safe to
run repeatedly (e.g. to pick up stragglers a transient failure missed).
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

from rich.console import Console

from . import db, dictionary, localdict
from .model import Candidate, Occurrence, normalize_pos
from .output import VOCAB_COLUMNS

# Small courtesy pause between lookups so we don't trip rate limits before the
# retry/backoff in dictionary._get even has to kick in.
_POLITE_DELAY = 0.15

_POS_TO_TAGGER = {"noun": "NOUN", "verb": "VERB", "adj": "ADJ", "adjective": "ADJ",
                  "adv": "ADV", "adverb": "ADV"}


def _row_to_candidate(row: dict) -> Candidate:
    pos = _POS_TO_TAGGER.get((row.get("part_of_speech") or "").lower(), "")
    cand = Candidate(lemma=row["word"], pos=pos)
    # Give the sense-picker the book sentence it needs.
    if row.get("sentence"):
        cand.occurrences.append(Occurrence(
            sentence=row["sentence"], chapter=row.get("chapter", ""),
            surface=row.get("as_seen") or row["word"],
        ))
    return cand


def _candidate_to_row(row: dict, cand: Candidate) -> dict:
    row["definition"] = cand.definition
    if cand.part_of_speech:
        row["part_of_speech"] = normalize_pos(cand.part_of_speech)
    if cand.ipa:
        row["ipa"] = cand.ipa
    if cand.synonyms:
        row["synonyms"] = "; ".join(cand.synonyms)
    if cand.etymology:
        row["etymology"] = cand.etymology
    if cand.definition_source:
        row["source"] = cand.definition_source
    return row


def refill(path: Path, console: Console | None = None) -> tuple[int, int]:
    """Enrich blank-definition rows in ``path`` in place. Returns (filled, attempted)."""
    console = console or Console()
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    todo = [r for r in rows if not (r.get("definition") or "").strip()]
    console.print(f"{len(rows)} rows · [bold]{len(todo)}[/bold] missing a definition.")
    if not todo:
        return 0, 0

    conn = db.connect()
    lexicon = localdict.build_lexicon(conn, {(r.get("word") or "").strip().lower() for r in todo})
    conn.close()

    session = dictionary.make_session()
    filled = 0
    with console.status("[bold]Looking up definitions…") as status:
        for i, row in enumerate(todo, 1):
            cand = _row_to_candidate(row)
            if not localdict.enrich(cand, lexicon):
                dictionary.enrich(cand, session)
            if cand.definition:
                _candidate_to_row(row, cand)
                filled += 1
            status.update(f"[bold]Looking up definitions… {i}/{len(todo)} · filled {filled}")
            time.sleep(_POLITE_DELAY)

    written = _write_rows(path, rows, console)
    console.print(f"[green]✓[/green] filled [bold]{filled}[/bold]/{len(todo)} · wrote {written.name}")
    return filled, len(todo)


def _write_rows(path: Path, rows: list[dict], console: Console) -> Path:
    """Write via a temp file + atomic replace. If the destination is locked (a
    Windows/drvfs lock — e.g. the CSV is open in Excel), fall back to an
    ``.enriched.csv`` sibling rather than losing a run's worth of lookups."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=VOCAB_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    try:
        os.replace(tmp, path)
        return path
    except PermissionError:
        fallback = path.with_suffix(".enriched.csv")
        os.replace(tmp, fallback)
        console.print(
            f"[yellow]![/yellow] {path.name} is locked (open in another program?) — "
            f"wrote [bold]{fallback.name}[/bold] instead."
        )
        return fallback


def main(argv: list[str] | None = None) -> int:
    import psycopg

    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    path = Path(argv[0])
    if not path.exists():
        print(f"no such file: {path}")
        return 1
    try:
        refill(path)
    except (RuntimeError, psycopg.Error) as exc:
        print(f"cannot connect to Postgres: {exc}")
        print("the local dictionary lookup needs vocab.wiktionary — "
              "set DATABASE_URL in the environment or a .env file")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
