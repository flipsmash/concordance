#!/usr/bin/env python3
"""One-time backfill: stamp vocab1_import=true on words that were already
imported from vocab.defined by import_defined_words() before that flag
existed (concordance ran it 2026-07-22 and 2026-07-24, ~8379 words, back
when the only provenance signal was "book-less + first_added=that date").

definition_source is untouched -- these words already carry vocab.defined's
own per-row source label (datamuse/phrontistery/oed/mw/...), which stays as
the more informative value; vocab1_import is purely an additional flag.

Discriminator: book-less (no word_book row) AND first_added on one of the
two known import dates AND lemma_lc matches a vocab.defined term. The date
filter is required, not just belt-and-suspenders -- book-less alone would
also catch words added via the unrelated admin "suggest a new word" flow
(webapp/backend/suggest_word.py) if one happens to match a vocab.defined
term by coincidence.

Usage:
    python scripts/backfill_vocab1_import_flag.py           # dry run (counts only)
    python scripts/backfill_vocab1_import_flag.py --apply   # actually update
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concordance import db  # noqa: E402

_IMPORT_DATES = ("2026-07-22", "2026-07-24")

_PREDICATE = f"""
    NOT EXISTS (SELECT 1 FROM {{s}}.word_book wb WHERE wb.word_id = w.id)
    AND w.first_added IN ({",".join("%s" for _ in _IMPORT_DATES)})
    AND EXISTS (SELECT 1 FROM vocab.defined d WHERE lower(d.term) = w.lemma_lc)
    AND NOT w.vocab1_import
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default=db.DEFAULT_SCHEMA)
    ap.add_argument("--apply", action="store_true", help="Actually run the UPDATE (default is a dry-run count).")
    args = ap.parse_args()
    s = args.schema

    conn = db.connect()
    db.apply_schema(conn, s)  # adds vocab1_import/vocab1_import_at if not already present
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {s}.word w WHERE {_PREDICATE.format(s=s)}", _IMPORT_DATES)
        matched = cur.fetchone()[0]

    print(f"matched {matched} word(s) eligible for vocab1_import=true")
    if matched != 8379:
        print(f"WARNING: expected exactly 8379 (98 on 2026-07-22 + 8281 on 2026-07-24) -- "
              f"got {matched}. Stop and investigate before --apply.")

    if not args.apply:
        print("dry run only -- pass --apply to update")
        conn.close()
        return

    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {s}.word w SET vocab1_import = true, vocab1_import_at = now() "
            f"WHERE {_PREDICATE.format(s=s)}",
            _IMPORT_DATES)
        updated = cur.rowcount
    conn.commit()
    conn.close()
    print(f"updated {updated} word(s)")


if __name__ == "__main__":
    main()
