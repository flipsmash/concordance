#!/usr/bin/env python3
"""Backlog correction for over-redacted quiz definitions (§ quizzing).

redact() (concordance/quizdef.py) is the fallback when an LLM rewrite still
leaks -- it blanks every leaking token, which can gut a short, templated
definition down to its scaffolding: "silkman" -> "A male dealer in silk."
becomes "A male dealer in —." -- grammatically intact but no longer
distinguishes silkman from any other "male dealer in X" trade word.

Two fixes landed together for this:
  1. quizdef.quizzable() now excludes any word whose quiz_definition is a
     too-sparse redaction (redaction_too_sparse) -- this alone stops any
     currently-bad OR future-bad over-redaction from ever being served in a
     quiz, going forward AND retroactively once compute_quizzable re-runs.
  2. Rewriter.rewrite() now escalates to a second, differently-strategized
     LLM prompt (_SYSTEM_HARD -- describe the broader category instead of
     dancing around the word) for anything that would otherwise fall to a
     too-sparse redaction, before accepting that fallback. This is forward-
     only in the pipeline; this script applies it to the EXISTING backlog.

This script: finds every word whose stored quiz_definition is a too-sparse
redaction, re-runs them through the (now-escalating) Rewriter using their
original `definition`, and writes back whatever it gets -- a real recovered
rewrite for some, an unchanged (still-sparse) redaction for the rest. Then
re-runs compute_quizzable so the ones that couldn't be recovered are
properly excluded rather than silently served.

Usage:
    python scripts/fix_over_redacted_definitions.py           # full backlog
    python scripts/fix_over_redacted_definitions.py --limit 200   # a sample first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concordance import db, quizdef  # noqa: E402
from concordance.config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default=db.DEFAULT_SCHEMA)
    ap.add_argument("--limit", type=int, default=0, help="Cap how many words to process (0 = all).")
    args = ap.parse_args()

    conn = db.connect()
    s = args.schema

    with conn.cursor() as cur:
        cur.execute(f"""SELECT id, lemma, definition, quiz_definition FROM {s}.word
                        WHERE quiz_def_source = 'redacted'""")
        rows = cur.fetchall()

    too_sparse = [(wid, lemma, defn, qd) for wid, lemma, defn, qd in rows
                  if quizdef.redaction_too_sparse(qd or "")]
    print(f"{len(too_sparse)} of {len(rows)} redacted definitions are too sparse to be a usable clue.")
    if args.limit:
        too_sparse = too_sparse[:args.limit]
        print(f"Processing first {len(too_sparse)} (--limit).")

    if not too_sparse:
        conn.close()
        return

    print("Loading the rewriter model (this can take a bit)...")
    rw = quizdef.Rewriter(Config())
    items = [{"word": lemma, "definition": defn} for _, lemma, defn, _ in too_sparse]
    result = rw.rewrite(items)

    recovered = 0
    with conn.cursor() as cur:
        for wid, lemma, defn, old_qd in too_sparse:
            new_qd, src = result[lemma.lower()]
            cur.execute(f"UPDATE {s}.word SET quiz_definition=%s, quiz_def_source=%s WHERE id=%s",
                        (new_qd, src, wid))
            if src == "rewritten":
                recovered += 1
    conn.commit()
    print(f"{recovered} of {len(too_sparse)} recovered with a real rewrite; "
          f"{len(too_sparse) - recovered} remain redacted (will be excluded from quizzing below).")

    print("Re-running compute_quizzable so any still-sparse ones are excluded...")
    dist = db.compute_quizzable(conn, s)
    print(f"quizzable distribution: {dist}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
