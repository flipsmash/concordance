#!/usr/bin/env python3
"""Online-cascade follow-up for scripts/fix_abbreviation_of_definitions.py.

That script resolved every "abbreviation of X" stub it could against the
local Wiktionary dump (vocab.wiktionary) -- 1,779 of 2,102. The remaining
~323 are stuck specifically because X itself isn't in that ~500k-term local
snapshot, not because the relationship is bogus: chimneypot ("abbreviation
of chimney pot") is a real, ordinary English compound, the local dump is
just missing a "chimney pot" entry. This script re-tries those targets
through the full online dictionary cascade (Free Dictionary API -> online
Wiktionary -> Merriam-Webster -> Wordnik -> yourdictionary.com, same tiers
`concordance deepen`/`concordance refill` use) instead of the local dump
alone.

Targets are deduplicated before querying (several headwords can point at
the same target -- calash/caleche both -> calèche) both to save time and
to stay well under Wordnik's 5-req/min free-tier cap.

Whatever's still unresolved after this keeps its original raw stub text
and its variant_flag_reason='abbreviation_stub' review flag untouched --
same never-guess, never-prune policy as the local-only pass. A word that
*does* get fixed here has that flag cleared (only if it's specifically
'abbreviation_stub' -- a foreign_language/misspelling flag from
sweep_variant_rejects.py is left alone).

Dry-run by default. Pass --apply to write changes.

Usage:
    python scripts/deepen_abbreviation_of_definitions.py                # dry run
    python scripts/deepen_abbreviation_of_definitions.py --limit 40     # sample
    python scripts/deepen_abbreviation_of_definitions.py --max-tier MW  # skip Wordnik/yourdict
    python scripts/deepen_abbreviation_of_definitions.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from concordance import db, deepdef, dictionary, localdict, mw, resolve  # noqa: E402
from concordance.model import Candidate  # noqa: E402

# resolve.resolve_definition's Tier.FREE step (dictionary.enrich) tries
# api.dictionaryapi.dev BEFORE falling back to Wiktionary -- and that host is
# unreachable from this environment (TLS handshake stalls, confirmed with
# curl outside Python too), costing ~30-70s of retry/backoff PER WORD before
# it ever reaches a host that actually answers. en.wiktionary.org, MW,
# Wordnik and yourdictionary.com all respond fine. So this script runs its
# own small cascade instead of resolve.resolve_definition, calling the
# online-Wiktionary/MW/Wordnik/yourdictionary tiers directly and skipping
# the Free Dictionary API step entirely.


def _lookup_online(lemma: str, session, wordnik_key: str, mw_api_key: str, max_tier: "resolve.Tier"):
    """Returns (gloss, source_tier_name) or (None, None)."""
    cand = Candidate(lemma=lemma, pos="")
    if dictionary._from_wiktionary(cand, session) and cand.definition:
        return cand.definition, "Wiktionary (online)"
    if max_tier >= resolve.Tier.MW and mw_api_key:
        if resolve._from_mw(cand, session, mw_api_key) and cand.definition:
            return cand.definition, "MW"
    if max_tier >= resolve.Tier.WORDNIK and wordnik_key:
        resolve._pace_wordnik()
        if deepdef._from_wordnik(cand, session, wordnik_key) and cand.definition:
            return cand.definition, "Wordnik"
    if max_tier >= resolve.Tier.YOURDICT:
        if deepdef._from_yourdictionary(cand, session) and cand.definition:
            return cand.definition, "yourdictionary.com"
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default=db.DEFAULT_SCHEMA)
    ap.add_argument("--limit", type=int, default=0, help="Cap how many stub rows to consider (0 = all).")
    ap.add_argument("--max-tier", default="YOURDICT", choices=[t.name for t in resolve.Tier],
                     help="Highest online tier to try (default YOURDICT; WEB needs a local LLM, skipped by default).")
    ap.add_argument("--apply", action="store_true", help="Actually write changes (default: dry run).")
    ap.add_argument("--database-url", default=None)
    args = ap.parse_args()

    console = Console()
    conn = db.connect(args.database_url)
    s = db._safe_schema(args.schema)
    max_tier = resolve.Tier[args.max_tier]

    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma, definition FROM {s}.word
                WHERE active AND definition ILIKE 'abbreviation of %' AND definition NOT LIKE '%—%'
                ORDER BY id""" + (f" LIMIT {int(args.limit)}" if args.limit else ""))
        rows = cur.fetchall()
    console.print(f"[bold]{len(rows)}[/bold] words still holding a raw 'abbreviation of' stub.")
    if not rows:
        conn.close()
        return

    targets: dict[str, list[tuple[int, str, str]]] = {}
    skipped_unparsed = 0
    for wid, lemma, old_def in rows:
        target = localdict.stub_target_phrase(old_def)
        if not target:
            skipped_unparsed += 1
            continue
        targets.setdefault(target.lower(), []).append((wid, lemma, old_def))

    console.print(f"  {len(targets)} distinct target phrases to look up online "
                  f"({skipped_unparsed} rows had unparseable stub text, left alone).")
    console.print(f"  max tier: [bold]{max_tier.name}[/bold] "
                  f"(Wordnik is rate-limited to 5/min, so a long tail can take a while).")

    session = dictionary.make_session()
    wordnik_key = deepdef.wordnik_key()
    mw_api_key = mw.mw_api_key()

    resolved_gloss: dict[str, str] = {}
    with console.status("[bold]Querying online sources…") as status:
        for i, target in enumerate(sorted(targets), 1):
            gloss, source = _lookup_online(target, session, wordnik_key, mw_api_key, max_tier)
            if gloss:
                resolved_gloss[target] = gloss
            status.update(f"[bold]Querying online sources… {i}/{len(targets)} "
                          f"({len(resolved_gloss)} found, last: {target} -> {source})")

    console.print(f"  online lookups succeeded for [bold]{len(resolved_gloss)}[/bold] / {len(targets)} distinct targets.")

    resolved_rows: list[tuple[int, str, str, str]] = []  # id, lemma, old_def, new_def
    for target, gloss in resolved_gloss.items():
        for wid, lemma, old_def in targets[target]:
            target_raw = localdict.stub_target_phrase(old_def)
            new_def = localdict.compose_stub_replacement(lemma, target_raw, gloss)
            if new_def:
                resolved_rows.append((wid, lemma, old_def, new_def))

    console.print(f"  words fixed: [bold]{len(resolved_rows)}[/bold] / {len(rows)}")
    console.print("\n[dim]Sample — fixed via online cascade (up to 25):[/dim]")
    for _, lemma, old_def, new_def in resolved_rows[:25]:
        console.print(f"  {lemma}: [strike]{old_def}[/strike] -> {new_def}")

    still_stuck = len(rows) - len(resolved_rows)
    console.print(f"\n[yellow]{still_stuck}[/yellow] words remain unresolved even after the online pass "
                  f"-- left as-is, still flagged for review.")

    if not args.apply:
        console.print("\n[yellow]Dry run — no changes made. Re-run with --apply to write them.[/yellow]")
        conn.close()
        return

    with conn.cursor() as cur:
        for i, (wid, _lemma, _old_def, new_def) in enumerate(resolved_rows, 1):
            cur.execute(
                f"""UPDATE {s}.word
                        SET definition=%s, quiz_definition=%s, quiz_def_source='clean',
                            variant_flag_reason=CASE WHEN variant_flag_reason='abbreviation_stub'
                                                      THEN NULL ELSE variant_flag_reason END,
                            variant_flag_note=CASE WHEN variant_flag_reason='abbreviation_stub'
                                                    THEN NULL ELSE variant_flag_note END,
                            variant_flagged_at=CASE WHEN variant_flag_reason='abbreviation_stub'
                                                     THEN NULL ELSE variant_flagged_at END,
                            updated_at=now()
                    WHERE id=%s""",
                (new_def, new_def, wid))
            if i % 200 == 0:
                conn.commit()
        conn.commit()

    console.print(f"\n[green]✓[/green] rewrote [bold]{len(resolved_rows)}[/bold] definitions via the online cascade.")

    console.print("\nRecomputing quizzable flags (some quiz_definitions changed)...")
    dist = db.compute_quizzable(conn, s)
    console.print(f"quizzable distribution: {dist}")

    conn.close()


if __name__ == "__main__":
    main()
