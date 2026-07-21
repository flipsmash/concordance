#!/usr/bin/env python3
"""Read-only audit for the filtering fixes landed 2026-07-20 (judge.py's
UNSURE veto + hint-forwarding, tokenize.py's ALL-CAPS cap_ratio exclusion,
pipeline.py's junk-POS cast-out on every re-encounter).

Every fix so far is forward-only -- none of them touch data already sitting
in the database. This script never writes anything; it only counts and
samples what each fix WOULD flip, so that decision can be made deliberately
rather than by just kicking off an expensive re-processing pass.

Three sections, in increasing order of how expensive it would be to actually
act on them:

  A. Fix 4 (junk-POS cast-out) -- EXACT. A currently-active word whose
     stored part_of_speech is already 'proper noun'/'symbol' is definitely
     wrong under the new code (a fresh encounter would cast it out
     immediately) -- no recomputation needed, just a query. Safe to act on
     directly.

  B. Fix 3 (ALL-CAPS cap_ratio) -- TRIAGE, not a verdict. Precisely
     re-deciding a rejected word's cap_ratio needs the original book text
     re-tokenized (cap_ratio is an aggregate over every occurrence, not
     something stored per-word) -- a real, large batch job (11k+ archived
     books), not run here. Instead: how many DISTINCT proper_noun-rejected
     lemmas are themselves in a trusted dictionary authority -- the same
     population an ALL-CAPS false positive would be drawn from. This will
     also catch genuine Bloom/Baker collisions (a real name that's also a
     dictionary word) as false leads -- it narrows where to look, it does
     not tell you which ones were wrong.

  C. Fix 1+2 (UNSURE veto + judge hint) -- TRIAGE, not a verdict. Re-running
     validity.py's logic against CURRENT word data (zipf, an approximated
     occurrence count from word_book link count, the word's current stored
     representative sentence) identifies which active words would land on
     UNSURE today. Whether the fixed judge would actually drop them needs a
     real LLM call per candidate -- opt-in via --judge, off by default so a
     normal run stays fast and free of model load time.

Usage:
    python scripts/audit_filtering_fixes.py                  # sections A+B+C, no LLM
    python scripts/audit_filtering_fixes.py --judge           # also runs the LLM on C's candidates
    python scripts/audit_filtering_fixes.py --judge --limit 200   # cap the LLM pass for a quick look
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concordance import db, localdict  # noqa: E402
from concordance.config import Config  # noqa: E402
from concordance.model import Candidate, Occurrence, Verdict, junk_pos_reason  # noqa: E402
from concordance.validity import ValidityGate  # noqa: E402


def section_a_junk_pos_active(conn, schema: str) -> list[tuple[str, str]]:
    print("\n=== A. Currently-active words with a junk POS (Fix 4, exact) ===")
    with conn.cursor() as cur:
        cur.execute(f"SELECT lemma, part_of_speech FROM {schema}.word WHERE active")
        rows = cur.fetchall()
    flagged = [(lemma, pos) for lemma, pos in rows if junk_pos_reason(pos)]
    print(f"{len(flagged)} of {len(rows)} active words have a junk POS stored right now.")
    if flagged:
        print("Sample (up to 20):")
        for lemma, pos in flagged[:20]:
            print(f"  {lemma!r} -> {pos!r}")
        print("\nThese are unambiguous under the new code -- the next time any of them is")
        print("re-encountered in an ingest, they'll be cast out automatically. To act on")
        print("them NOW without waiting for a re-encounter, they'd need a direct")
        print("`UPDATE word SET active=false WHERE lemma = ANY(...)` for this exact list.")
    return flagged


def section_b_proper_noun_reject_triage(conn, schema: str) -> list[str]:
    print("\n=== B. Dictionary-word lemmas among proper_noun rejections (Fix 3, triage) ===")
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT lemma FROM {schema}.rejected_word WHERE reason = 'proper_noun'")
        lemmas = [r[0] for r in cur.fetchall()]
    print(f"{len(lemmas)} distinct lemmas ever rejected as proper_noun.")

    gate = ValidityGate(Config(), local_dict={})
    local_dict = localdict.build_lexicon(conn, set(lemmas))

    candidates = []
    for i, lemma in enumerate(lemmas, 1):
        if lemma in local_dict or lemma in gate._sym.words or gate._in_wordnet(lemma) or lemma in gate._words:
            candidates.append(lemma)
        if i % 50_000 == 0:
            print(f"  ...checked {i}/{len(lemmas)}")

    print(f"{len(candidates)} of those are ALSO in a trusted dictionary authority "
          f"(local Wiktionary / 82k wordlist / WordNet / NLTK 234k) --")
    print("the same population an ALL-CAPS false positive would be drawn from.")
    print("NOT a verdict: this also contains genuine Bloom/Baker collisions (a real name")
    print("that happens to also be a dictionary word) as false leads. Getting an actual")
    print("per-word answer needs the original archived book text re-tokenized (cap_ratio")
    print("is an aggregate over every occurrence in a book, not stored per-word) --")
    print("a real batch job across the archive/ directory, not done here.")
    if candidates:
        print("Sample (up to 20):")
        for lemma in candidates[:20]:
            print(f"  {lemma!r}")
    return candidates


def section_c_unsure_candidates(conn, schema: str, run_judge: bool, limit: int) -> list[dict]:
    print("\n=== C. Active words that would land on UNSURE today (Fix 1+2, triage) ===")
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT w.id, w.lemma, w.sentence, w.part_of_speech,
                   (SELECT count(*) FROM {schema}.word_book wb WHERE wb.word_id = w.id) AS book_count
            FROM {schema}.word w WHERE w.active
        """)
        rows = cur.fetchall()
    print(f"Re-checking {len(rows)} active words against validity.py's current logic "
          "(no DB writes, no LLM yet)...")

    lemmas = {r[1] for r in rows}
    local_dict = localdict.build_lexicon(conn, lemmas)
    gate = ValidityGate(Config(), local_dict=local_dict)

    unsure = []
    for i, (word_id, lemma, sentence, pos, book_count) in enumerate(rows, 1):
        from wordfreq import zipf_frequency
        # word_book link count (distinct BOOKS) is a coarse proxy for
        # validity.py's real `count` (occurrences WITHIN one book's
        # candidate, capped at 12) -- not the same measurement, but a
        # reasonable stand-in for "does this recur enough to be a possible
        # coinage rather than a one-off," which is all coinage_min_count
        # actually gates on.
        occs = [Occurrence(sentence=sentence or "", chapter="", surface=lemma)] * max(book_count, 1)
        c = Candidate(lemma=lemma, pos=pos or "NOUN", occurrences=occs, zipf=zipf_frequency(lemma, "en"))
        gate.judge(c)
        if c.verdict is Verdict.UNSURE:
            unsure.append({"id": word_id, "lemma": lemma, "reason": c.interesting_reason})
        if i % 10_000 == 0:
            print(f"  ...checked {i}/{len(rows)}")

    print(f"{len(unsure)} active words would land on UNSURE under current validity.py logic.")
    if unsure:
        print("Sample (up to 20):")
        for u in unsure[:20]:
            print(f"  {u['lemma']!r} -- {u['reason']}")
        print("\nWhether the FIXED judge (Fix 1+2) would actually drop each of these needs a")
        print("real model call -- not run here unless --judge is passed.")

    if run_judge and unsure:
        from concordance.judge import get_judge

        batch = unsure[:limit] if limit else unsure
        print(f"\n--judge passed: running the live judge on {len(batch)} of {len(unsure)} candidates...")
        cfg = Config()
        judge = get_judge(cfg)
        cands = []
        for u in batch:
            c = Candidate(lemma=u["lemma"], pos="NOUN")
            c.verdict = Verdict.UNSURE
            c.interesting_reason = u["reason"]
            cands.append(c)
        judge.judge(cands)
        would_drop = [c for c in cands if c.verdict is Verdict.DROP]
        print(f"{len(would_drop)} of {len(batch)} would be dropped by the fixed judge right now.")
        if would_drop:
            print("Sample (up to 30):")
            for c in would_drop[:30]:
                print(f"  {c.lemma!r}")
        print("\nThese ARE actionable: re-running `concordance ingest`/`maintain` won't touch them")
        print("(the cross-book verdict cache treats them as already-known), so acting on this list")
        print("needs a deliberate re-judge pass that bypasses the cache for exactly these lemmas.")

    return unsure


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default=db.DEFAULT_SCHEMA)
    ap.add_argument("--judge", action="store_true",
                     help="Also run the live LLM judge on section C's candidates (slow, loads the model).")
    ap.add_argument("--limit", type=int, default=0,
                     help="Cap how many of section C's candidates get sent to the judge (0 = all).")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--skip-c", action="store_true")
    args = ap.parse_args()

    conn = db.connect()
    print(f"Auditing schema {args.schema!r} -- read-only, no writes will be made.")

    if not args.skip_a:
        section_a_junk_pos_active(conn, args.schema)
    if not args.skip_b:
        section_b_proper_noun_reject_triage(conn, args.schema)
    if not args.skip_c:
        section_c_unsure_candidates(conn, args.schema, run_judge=args.judge, limit=args.limit)

    conn.close()
    print("\nDone. No changes were made to the database.")


if __name__ == "__main__":
    main()
