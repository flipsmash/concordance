#!/usr/bin/env python3
"""Backlog correction for bogus "abbreviation of X" definitions.

Brian's report: some words store a definition of the literal form
"abbreviation of X." that doesn't make sense as an abbreviation -- e.g.
"waterfit" -> "abbreviation of aquafitness." (the two words share almost no
letters). Investigation found the label is untrustworthy across the board:
concordance/localdict.py copies `vocab.wiktionary.definition` straight
through with no content check, and that dump uses the same "abbreviation
of X" phrasing for real clippings (contemp -> contemporary), plain spelling
variants (vizor -> visor), compound-spacing variants (waterpower -> water
power), and case-only variants (churrigueresque -> Churrigueresque) alike --
there's no way to tell which is which from the dump's text alone.

Fix (concordance/localdict.py, now applied going forward at every
enrichment call site via localdict.expand_lexicon_for_stubs +
localdict.resolve_stub_definition): ignore the "abbreviation" label,
resolve X's own real definition from the same local Wiktionary dump
(following a resolution chain if X is itself a stub, folding accents,
trying the compound both closed and open), and render an honest
replacement -- "Abbreviation of X -- <X's real definition>" when the
headword really does look like a truncation of a longer X, "Variant
spelling of X -- <X's real definition>" otherwise. Both phrasings still
trip quizdef._VARIANT_RE, so quiz-suitability is unaffected. Words where X
isn't resolvable in the local dump at all (~5% in the real-data sample --
corrupted source markup, technical acronym expansions with no headword of
their own, or a target simply missing from the dump) are left as-is and
flagged via variant_flag_reason for human review, never silently pruned or
guessed at.

This script applies that same fix to the backlog: every currently-active
word whose definition already reads "abbreviation of ...".

Dry-run by default: prints counts + samples of what WOULD change. Pass
--apply to write definition + quiz_definition (kept in sync -- both held
the identical raw stub text) and variant_flag_reason/_note for the rest.

Usage:
    python scripts/fix_abbreviation_of_definitions.py                # dry run
    python scripts/fix_abbreviation_of_definitions.py --limit 50     # sample
    python scripts/fix_abbreviation_of_definitions.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from concordance import db, localdict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default=db.DEFAULT_SCHEMA)
    ap.add_argument("--limit", type=int, default=0, help="Cap how many words to process (0 = all).")
    ap.add_argument("--apply", action="store_true", help="Actually write changes (default: dry run).")
    ap.add_argument("--database-url", default=None)
    args = ap.parse_args()

    console = Console()
    conn = db.connect(args.database_url)
    s = db._safe_schema(args.schema)

    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma, definition, quiz_definition FROM {s}.word
                WHERE active AND definition ILIKE 'abbreviation of %'
                ORDER BY id""" + (f" LIMIT {int(args.limit)}" if args.limit else ""))
        rows = cur.fetchall()
    console.print(f"[bold]{len(rows)}[/bold] active words with an 'abbreviation of' definition.")
    if not rows:
        conn.close()
        return

    lexicon = localdict.build_lexicon(conn, {lemma.lower() for _, lemma, _, _ in rows})
    # Gather targets straight from each row's own stored definition, not
    # just from what expand_lexicon_for_stubs finds already in `lexicon` --
    # a handful of these words were originally defined via the *online*
    # Wiktionary API, so their own headword was never in the local dump at
    # all and wouldn't otherwise surface as something needing a target fetch.
    needed = {k for _, _, old_def, _ in rows for k in localdict.stub_target_candidates(old_def)}
    lexicon.update({k: v for k, v in localdict.build_lexicon(conn, needed).items() if k not in lexicon})
    localdict.expand_lexicon_for_stubs(conn, lexicon)

    resolved: list[tuple[int, str, str, str]] = []  # id, lemma, old_def, new_def
    unresolved: list[tuple[int, str, str]] = []  # id, lemma, old_def
    for wid, lemma, old_def, _quiz_def in rows:
        new_def = localdict.resolve_stub_definition(
            lemma, old_def, lambda k: localdict._best_entry(lexicon.get(k)))
        if new_def:
            resolved.append((wid, lemma, old_def, new_def))
        else:
            unresolved.append((wid, lemma, old_def))

    console.print(f"  resolved to real content: [bold]{len(resolved)}[/bold]")
    console.print(f"  unresolvable (flagged for review): [bold]{len(unresolved)}[/bold]")

    console.print("\n[dim]Sample — resolved (up to 20):[/dim]")
    for _, lemma, old_def, new_def in resolved[:20]:
        console.print(f"  {lemma}: [strike]{old_def}[/strike] -> {new_def}")
    console.print("\n[dim]Sample — unresolvable, will be flagged not changed (up to 20):[/dim]")
    for _, lemma, old_def in unresolved[:20]:
        console.print(f"  {lemma}: {old_def}")

    if not args.apply:
        console.print("\n[yellow]Dry run — no changes made. Re-run with --apply to write them.[/yellow]")
        conn.close()
        return

    with conn.cursor() as cur:
        for i, (wid, _lemma, _old_def, new_def) in enumerate(resolved, 1):
            # quiz_def_source='clean' alongside quiz_definition: whatever was
            # there before (an LLM 'rewritten' paraphrase, or a 'redacted'
            # one) was a rewrite of the now-replaced bogus stub text, so it's
            # stale regardless -- leaving the old source label would lie
            # about provenance once quiz_definition holds this resolved text
            # verbatim instead. Both phrasings still trip quizdef._VARIANT_RE
            # so this doesn't change quiz-eligibility either way.
            cur.execute(
                f"""UPDATE {s}.word
                        SET definition=%s, quiz_definition=%s, quiz_def_source='clean', updated_at=now()
                    WHERE id=%s""",
                (new_def, new_def, wid))
            if i % 500 == 0:
                conn.commit()
        conn.commit()
        for i, (wid, lemma, old_def) in enumerate(unresolved, 1):
            cur.execute(
                f"""UPDATE {s}.word
                        SET variant_flag_reason=COALESCE(variant_flag_reason, %s),
                            variant_flag_note=COALESCE(variant_flag_note, %s),
                            variant_flagged_at=COALESCE(variant_flagged_at, now()),
                            updated_at=now()
                    WHERE id=%s""",
                ("abbreviation_stub",
                 f'Local Wiktionary only has "{old_def}" -- target not resolvable in the local dump.',
                 wid))
            if i % 500 == 0:
                conn.commit()
        conn.commit()

    console.print(f"\n[green]✓[/green] rewrote [bold]{len(resolved)}[/bold] definitions, "
                  f"flagged [bold]{len(unresolved)}[/bold] for human review.")

    console.print("\nRecomputing quizzable flags (some quiz_definitions changed)...")
    dist = db.compute_quizzable(conn, s)
    console.print(f"quizzable distribution: {dist}")

    conn.close()


if __name__ == "__main__":
    main()
