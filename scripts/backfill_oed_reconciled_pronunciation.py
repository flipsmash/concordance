#!/usr/bin/env python3
"""One-time catch-up for oed.entry rows whose vision-LLM double-pass
transcription was already computed (pronunciation_pass1/pass2 both stored)
but discarded because the old resolve_pronunciation() required byte-exact
agreement -- see concordance/oed/pronunciation.py's _reconcile_close for the
two narrow, individually-justified equivalences it now also tries (ASCII
apostrophe as a stress glyph; stress-mark presence/absence on a
monosyllable). Measured live: 1,355 of 45,823 stuck entries resolve this
way, of which 54 are concordance IPA-gap words that gain usable pronunciation
(the rest either don't match a concordance word, or land on a headword that
already has a DIFFERENT resolved ipa elsewhere -- ambiguous homographs the
Tier 4 cascade discards regardless, per resolve_pronunciation.py's
_oed_ipa). This script exists only for that one-time catch-up: every NEW
entry oed/pipeline.py resolves going forward already goes through the same
(now-fixed) resolve_pronunciation, no separate backfill needed for those.

Dry-run by default: prints counts + a sample of what WOULD change.
Pass --apply to actually write pronunciation_ipa/pronunciation_source/
pronunciation_needs_review.

Usage:
    python scripts/backfill_oed_reconciled_pronunciation.py                # dry run
    python scripts/backfill_oed_reconciled_pronunciation.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from concordance import db  # noqa: E402
from concordance.oed import db as oed_db  # noqa: E402
from concordance.oed import pronunciation as p  # noqa: E402

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

    cur.execute(f"""
        SELECT id, headword, pronunciation_pass1, pronunciation_pass2, pronunciation_raw
        FROM {schema}.entry
        WHERE pronunciation_pass1 IS NOT NULL AND pronunciation_pass2 IS NOT NULL
          AND (pronunciation_ipa IS NULL OR pronunciation_ipa = '')
    """)
    rows = cur.fetchall()
    console.print(f"[bold]{len(rows)}[/bold] stuck entries with both passes stored")

    resolved = []
    for entry_id, headword, pass1, pass2, raw in rows:
        ipa, needs_review = p.resolve_pronunciation(pass1, pass2, raw)
        if ipa is not None:
            resolved.append((entry_id, headword, pass1, pass2, raw, ipa))

    console.print(f"[green]{len(resolved)}[/green] resolve via the narrow reconciliation rules")
    console.print("sample:")
    for entry_id, headword, pass1, pass2, raw, ipa in resolved[:15]:
        console.print(f"  {headword!r}: {pass1!r} / {pass2!r} -> {ipa!r}")

    if not args.apply:
        console.print("\n[yellow]dry run — pass --apply to write these[/yellow]")
        return

    for entry_id, headword, pass1, pass2, raw, ipa in resolved:
        oed_db.update_pronunciation(
            conn, entry_id, pronunciation_raw=raw, pass1=pass1, pass2=pass2,
            # 'vision_llm_reconciled' would be more precise provenance, but
            # entry_pronunciation_source_check only allows
            # {'vision_llm', 'manual'} -- this genuinely is vision_llm
            # output, just merged via the narrow rules instead of verbatim
            # agreement, and nothing downstream branches on the distinction.
            ipa=ipa, source="vision_llm", needs_review=False, schema=schema,
        )
    conn.commit()
    console.print(f"[green]✓[/green] wrote {len(resolved)} reconciled pronunciations")


if __name__ == "__main__":
    main()
