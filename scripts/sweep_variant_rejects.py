#!/usr/bin/env python3
"""Retroactive sweep for the foreign-word / archaic-spelling-variant gate
(validity_score.variant_reject_reason) against every word ALREADY active and
defined in the live schema -- these are words that got their KEEP verdict
before the gate existed (or via the cross-book verdict cache, which never
re-runs ingest's ValidityGate once a lemma has any cached verdict at all),
so the new gate wired into pipeline.py/fill_definitions never had a chance
to see them.

Dry-run by default: prints counts + a sample of what WOULD be cast out.
Pass --apply to actually flip active=false on the matches (same cast-out
shape fill_definitions itself uses: just active=false + updated_at, no
rejected_word row -- this isn't tied to any one book).

Usage:
    python scripts/sweep_variant_rejects.py                # dry run, samples
    python scripts/sweep_variant_rejects.py --apply         # actually cast out
    python scripts/sweep_variant_rejects.py --schema concordance --apply
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from concordance import db, validity_score  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schema", default=db.DEFAULT_SCHEMA)
    parser.add_argument("--apply", action="store_true", help="Actually cast out matches (default: dry run).")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    console = Console()
    conn = db.connect(args.database_url)
    s = db._safe_schema(args.schema)

    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma FROM {s}.word WHERE active AND coalesce(definition,'') <> '' ORDER BY id")
        rows = cur.fetchall()
    console.print(f"[bold]{len(rows)}[/bold] active, defined words to check.")

    t0 = time.monotonic()
    foreign_hits: list[tuple[int, str, str]] = []
    variant_hits: list[tuple[int, str, str]] = []
    with console.status("[bold]Scanning…") as status:
        for i, (wid, lemma) in enumerate(rows, 1):
            result = validity_score.variant_reject_reason(lemma)
            if result:
                reason, note = result
                if reason.value == "foreign_language":
                    foreign_hits.append((wid, lemma, note))
                else:
                    variant_hits.append((wid, lemma, note))
            if i % 2000 == 0:
                status.update(f"[bold]Scanning… {i}/{len(rows)}")
    elapsed = time.monotonic() - t0

    console.print(f"\n[bold]Scan complete in {elapsed:.1f}s[/bold] "
                  f"({len(rows) / elapsed:.0f} words/sec).")
    console.print(f"  foreign-language: [bold]{len(foreign_hits)}[/bold]")
    console.print(f"  archaic/OCR spelling variant: [bold]{len(variant_hits)}[/bold]")
    console.print(f"  total to cast out: [bold]{len(foreign_hits) + len(variant_hits)}[/bold]"
                  f" / {len(rows)} ({(len(foreign_hits) + len(variant_hits)) / max(len(rows), 1) * 100:.1f}%)")

    console.print("\n[dim]Sample — foreign (up to 20):[/dim]")
    for _, lemma, note in foreign_hits[:20]:
        console.print(f"  {lemma}: {note}")
    console.print("\n[dim]Sample — archaic/OCR spelling variant (up to 20):[/dim]")
    for _, lemma, note in variant_hits[:20]:
        console.print(f"  {lemma}: {note}")

    if not args.apply:
        console.print("\n[yellow]Dry run — no changes made. Re-run with --apply to cast these out.[/yellow]")
        conn.close()
        return

    all_hits = foreign_hits + variant_hits
    with conn.cursor() as cur:
        for i, (wid, lemma, _) in enumerate(all_hits, 1):
            cur.execute(f"UPDATE {s}.word SET active=false, updated_at=now() WHERE id=%s", (wid,))
            if i % 500 == 0:
                conn.commit()
    conn.commit()
    console.print(f"\n[green]✓[/green] cast out [bold]{len(all_hits)}[/bold] words.")
    conn.close()


if __name__ == "__main__":
    main()
