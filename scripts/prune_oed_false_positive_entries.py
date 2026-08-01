#!/usr/bin/env python3
"""Retroactive cleanup for oed.entry rows ingested before
parse.looks_like_definition_entry existed: ordinary body-text words the
size+margin headword heuristic (segment.py) wrongly flagged as headwords
(e.g. "form", "one", "great", "whence", "common") -- confirmed by the user
reviewing real output ("oppose", "form", "main", "great" showing up
repeatedly but non-contiguously). A real definition entry always has a
pronunciation bracket or POS abbreviation immediately after the headword;
these don't. See parse.looks_like_definition_entry's docstring for the
measured false-positive rate (39.6% of all 3,576 rows at time of writing).

DELETE, not a full re-ingest: only 16 of the 182 already-resolved
pronunciations (the expensive vision-model output) belong to entries that
fail this check, so a targeted delete preserves 166 of them instead of
re-paying that cost. definition/quotation rows cascade via FK.

Dry-run by default: prints counts + a sample of what WOULD be deleted.
Pass --apply to actually delete.

Usage:
    python scripts/prune_oed_false_positive_entries.py                # dry run
    python scripts/prune_oed_false_positive_entries.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from concordance import db  # noqa: E402
from concordance.oed.parse import looks_like_definition_entry  # noqa: E402

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
    cur.execute(
        f"select id, headword, raw_text, pronunciation_ipa is not null "
        f"from {schema}.entry"
    )
    rows = cur.fetchall()

    fail_ids, fail_with_ipa, fail_samples = [], 0, []
    for entry_id, headword, raw_text, has_ipa in rows:
        if not looks_like_definition_entry(headword, raw_text):
            fail_ids.append(entry_id)
            if has_ipa:
                fail_with_ipa += 1
            if len(fail_samples) < 40:
                fail_samples.append(headword)

    console.print(f"total entries: {len(rows)}")
    console.print(f"failing (to be deleted): {len(fail_ids)} "
                   f"({100 * len(fail_ids) / len(rows):.1f}%)")
    console.print(f"failing entries with an already-resolved pronunciation "
                   f"(would be lost): {fail_with_ipa}")
    console.print(f"sample of failing headwords: {fail_samples}")

    if not args.apply:
        console.print("[yellow]Dry run only -- pass --apply to delete.[/yellow]")
        return

    cur.execute(
        f"delete from {schema}.entry where id = any(%s)", (fail_ids,)
    )
    conn.commit()
    console.print(f"[green]Deleted {len(fail_ids)} entries.[/green]")


if __name__ == "__main__":
    main()
