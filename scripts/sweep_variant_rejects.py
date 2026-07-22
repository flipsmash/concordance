#!/usr/bin/env python3
"""Retroactive human-review flag for the foreign-word / archaic-spelling-
variant detectors (validity_score.variant_reject_reason) against every word
ALREADY active and defined in the live schema -- these are words that got
their KEEP verdict before the detectors existed (or via the cross-book
verdict cache, which never re-runs ingest's ValidityGate once a lemma has
any cached verdict at all), so the check wired into pipeline.py/
fill_definitions never had a chance to see them.

NOT an auto-reject: a real-scale dry run of this exact sweep found the
detectors flag ~21% of the live vocabulary with a false-positive rate far
too high to trust unattended (haft/glaive/thurible/discomfit all wrongly
flagged) -- see the commit that disabled the hard-gate wiring for the full
finding. This script only ever WRITES word.variant_flag_reason/_note/_at
(the same review-queue columns pipeline.py/fill_definitions/refill_definitions
set going forward) so a human can query, sample, and manually prune via the
review webapp -- it never touches word.active.

Dry-run by default: prints counts + a sample of what WOULD be flagged.
Pass --apply to actually write the flag columns.

Usage:
    python scripts/sweep_variant_rejects.py                # dry run, samples
    python scripts/sweep_variant_rejects.py --apply         # write the flags
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
    parser.add_argument("--apply", action="store_true", help="Actually write the flags (default: dry run).")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    console = Console()
    conn = db.connect(args.database_url)
    s = db._safe_schema(args.schema)

    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma FROM {s}.word "
                    f"WHERE active AND coalesce(definition,'') <> '' AND variant_flag_reason IS NULL "
                    f"ORDER BY id")
        rows = cur.fetchall()
    console.print(f"[bold]{len(rows)}[/bold] active, defined, not-yet-flagged words to check.")

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
                  f"({len(rows) / max(elapsed, 0.01):.0f} words/sec).")
    console.print(f"  foreign-language: [bold]{len(foreign_hits)}[/bold]")
    console.print(f"  archaic/OCR spelling variant: [bold]{len(variant_hits)}[/bold]")
    console.print(f"  total to flag: [bold]{len(foreign_hits) + len(variant_hits)}[/bold]"
                  f" / {len(rows)} ({(len(foreign_hits) + len(variant_hits)) / max(len(rows), 1) * 100:.1f}%)")

    console.print("\n[dim]Sample — foreign (up to 20):[/dim]")
    for _, lemma, note in foreign_hits[:20]:
        console.print(f"  {lemma}: {note}")
    console.print("\n[dim]Sample — archaic/OCR spelling variant (up to 20):[/dim]")
    for _, lemma, note in variant_hits[:20]:
        console.print(f"  {lemma}: {note}")

    if not args.apply:
        console.print("\n[yellow]Dry run — no changes made. Re-run with --apply to write the flags.[/yellow]")
        conn.close()
        return

    all_hits = ([(wid, lemma, note, "foreign_language") for wid, lemma, note in foreign_hits]
                + [(wid, lemma, note, "misspelling") for wid, lemma, note in variant_hits])
    with conn.cursor() as cur:
        for i, (wid, lemma, note, reason) in enumerate(all_hits, 1):
            cur.execute(
                f"""UPDATE {s}.word SET variant_flag_reason=%s, variant_flag_note=%s,
                        variant_flagged_at=now(), updated_at=now()
                    WHERE id=%s""",
                (reason, note, wid))
            if i % 500 == 0:
                conn.commit()
    conn.commit()
    console.print(f"\n[green]✓[/green] flagged [bold]{len(all_hits)}[/bold] words for human review.")
    console.print("[dim]Query: SELECT lemma, variant_flag_reason, variant_flag_note FROM "
                  f"{s}.word WHERE variant_flag_reason IS NOT NULL ORDER BY lemma;[/dim]")
    conn.close()


if __name__ == "__main__":
    main()
