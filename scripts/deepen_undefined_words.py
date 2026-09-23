#!/usr/bin/env python3
"""Deep definition-acquisition pass for the standing "never defined" backlog.

Brian's question: why are there still active, valid-looking words with no
definition at all, and can OED help? Investigation found: those words
*have* gone through the full cascade before, but their last check predates
this project's OED ingestion finishing (or is simply weeks stale) --
db.fill_definitions()'s 14-day recheck cooldown means nobody's re-asked OED,
Merriam-Webster, Wordnik, or the web-search+LLM tier since. This script is
just a driver for db.fill_definitions() (all the real logic -- cast-out for
junk POS, MW quota checks, validity_score bookkeeping, flag handling --
lives there and is reused as-is), with two things layered on:

1. Scoped to validity_label IN ('likely-valid', 'uncertain') by default --
   skips the 'likely-artifact' tail (probably OCR noise) per Brian's call,
   since burning WEB/LLM time on those is the worst ROI in the backlog.
2. Skips the Free Dictionary API tier (dictionary._from_freedict). That
   host (api.dictionaryapi.dev) is currently unreachable from this machine
   -- TLS handshake stalls, confirmed with curl outside Python too -- and
   dictionary.py's retry/backoff costs 30-70s of pure dead time per word
   before falling through. Across thousands of words that's potentially
   days wasted on a dead host. This is a monkeypatch, not a permanent
   change to dictionary.py -- that host may recover, and disabling it for
   every caller everywhere is a separate decision from "make today's run
   not hang."

Runs use_web=True (WEB tier: DuckDuckGo search + local LLM extraction) by
Brian's choice -- needs the configured model_path to exist (checked before
starting) and takes noticeably longer per word than the OED/MW/Wordnik/
yourdictionary tiers alone.

Usage:
    python scripts/deepen_undefined_words.py --limit 30       # bounded test run
    python scripts/deepen_undefined_words.py                  # full scoped backlog
    python scripts/deepen_undefined_words.py --all-labels      # don't skip likely-artifact
    python scripts/deepen_undefined_words.py --no-web          # skip the LLM/web-search tier
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from concordance import db, dictionary  # noqa: E402
from concordance.config import Config  # noqa: E402


def _skip_freedict(cand, session):
    """api.dictionaryapi.dev is unreachable from this machine right now --
    fail instantly instead of dictionary._get's 30-70s retry/backoff, so
    the cascade falls through to Wiktionary (which does answer) immediately."""
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default=db.DEFAULT_SCHEMA)
    ap.add_argument("--limit", type=int, default=0, help="Cap how many words to process (0 = all in scope).")
    ap.add_argument("--all-labels", action="store_true",
                     help="Don't restrict to likely-valid/uncertain -- include likely-artifact too.")
    ap.add_argument("--no-web", action="store_true", help="Skip the WEB (DuckDuckGo + local LLM) tier.")
    ap.add_argument("--recheck-after-days", type=int, default=14)
    ap.add_argument("--database-url", default=None)
    args = ap.parse_args()

    console = Console()
    use_web = not args.no_web
    if use_web:
        model_path = Config().model_path
        if not model_path or not Path(model_path).exists():
            console.print(f"[red]--no-web not passed, but no model file found at {model_path!r}. "
                           f"Pass --no-web or fix Config.model_path.[/red]")
            sys.exit(1)

    dictionary._from_freedict = _skip_freedict  # see module docstring

    conn = db.connect(args.database_url)
    validity_labels = None if args.all_labels else {"likely-valid", "uncertain"}

    console.print(f"[bold]Scope:[/bold] {'all validity labels' if args.all_labels else 'likely-valid + uncertain only'}"
                  f" · WEB tier: {'on' if use_web else 'off'} · limit: {args.limit or 'none'}")
    console.print("Running db.fill_definitions() -- this can take a while "
                   "(Wordnik is rate-limited to 5/min, WEB tier does a live search + LLM call per word)...")

    t0 = time.monotonic()
    stats = db.fill_definitions(
        conn, args.schema, limit=args.limit, use_web=use_web,
        recheck_after_days=args.recheck_after_days, validity_labels=validity_labels)
    elapsed = time.monotonic() - t0

    console.print(f"\n[green]Done in {elapsed/60:.1f} min.[/green]")
    console.print(f"  attempted: [bold]{stats['attempted']}[/bold]")
    console.print(f"  defined: [bold]{stats['defined']}[/bold]")
    console.print(f"  cast out (revealed as symbol/proper-noun/foreign-language junk): [bold]{stats['cast_out']}[/bold]")
    console.print(f"  still undefined (validity_score re-estimated, will retry again after "
                   f"{args.recheck_after_days}d): [bold]{stats['still_undefined']}[/bold]")

    conn.close()


if __name__ == "__main__":
    main()
