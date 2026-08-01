#!/usr/bin/env python3
"""Retroactive cleanup pass using concordance.oed.sequence.find_out_of_order_ids:
an entry whose headword breaks first-letter alphabetical order relative to
its neighbors (in reading order) is either leftover body-text noise that
slipped past looks_like_definition_entry's bracket/POS check by
coincidence, or an OCR-garbled headword (a stray extra letter glued onto
the front, e.g. "fabackstays" for "abackstays"). Both are pruned -- a
garbled headword isn't a trustworthy entry either, confirmed with Brian.

Runs per-volume (ordering only makes sense within one volume's own
alphabetical range).

Dry-run by default: prints counts + a sample of what WOULD be deleted.
Pass --apply to actually delete.

Usage:
    python scripts/prune_oed_out_of_order_entries.py                # dry run
    python scripts/prune_oed_out_of_order_entries.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from concordance import db  # noqa: E402
from concordance.oed.sequence import find_out_of_order_ids  # noqa: E402

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="oed")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    conn = db.connect(args.database_url)
    schema = db._safe_schema(args.schema)
    cur = conn.cursor()

    cur.execute(f"select id from {schema}.volume order by id")
    volume_ids = [r[0] for r in cur.fetchall()]

    total_deleted = 0
    for volume_id in volume_ids:
        cur.execute(
            f"select id, headword from {schema}.entry "
            f"where volume_id = %s order by id",
            (volume_id,),
        )
        rows = cur.fetchall()
        out_of_order = find_out_of_order_ids(rows)
        console.print(
            f"volume {volume_id}: {len(rows)} entries, "
            f"{len(out_of_order)} out of order ({100 * len(out_of_order) / max(len(rows), 1):.1f}%)"
        )
        samples = [hw for eid, hw in rows if eid in out_of_order][:30]
        console.print(f"  sample: {samples}")
        total_deleted += len(out_of_order)

        if args.apply and out_of_order:
            cur.execute(
                f"delete from {schema}.entry where id = any(%s)",
                (list(out_of_order),),
            )

    if args.apply:
        conn.commit()
        console.print(f"[green]Deleted {total_deleted} entries.[/green]")
    else:
        console.print(f"[yellow]Dry run only -- {total_deleted} would be deleted. Pass --apply to delete.[/yellow]")


if __name__ == "__main__":
    main()
