#!/usr/bin/env python3
"""Stand-alone CLI: look up a word on Merriam-Webster, print to stdout.

Two tiers, in order:

  1. The official Collegiate Dictionary API (concordance.mw) -- needs
     MW_DICTIONARY_API_KEY in .env or the environment. Cached on disk and
     quota-tracked so a batch of lookups can't blow through the free tier's
     1000 queries/day; a word already seen (hit OR miss) never costs another
     call.
  2. A Playwright scrape of the live site (concordance.mw_scrape) -- only
     tried when the API came back empty (miss or quota exhausted), for words
     the site has but the Collegiate API doesn't. Slower and more brittle
     (the site sits behind a Cloudflare bot challenge); see mw_scrape.py's
     docstring for how that's handled. Requires `pip install concordance[scrape]`
     and `playwright install chromium`.

Usage:
    python scripts/lookup_mw.py concordance
    python scripts/lookup_mw.py concordance run --headless
    python scripts/lookup_mw.py cangue --no-fallback
"""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402
from rich.console import Console  # noqa: E402

from concordance import mw  # noqa: E402

console = Console(soft_wrap=True)


def _print_entry(entry: "mw.MWEntry") -> None:
    console.print(f"[bold]{entry.headword}[/bold] [italic]{entry.part_of_speech}[/italic] [dim]({entry.source})[/dim]")
    for p in entry.pronunciations:
        bits = [p.respelling] if p.respelling else []
        if p.audio_url:
            bits.append(f"[dim]{p.audio_url}[/dim]")
        if bits:
            console.print(f"  {'  '.join(bits)}")
    for i, d in enumerate(entry.definitions, 1):
        console.print(f"  {i}. {d}")
    if entry.etymology:
        console.print(f"  [dim]Etymology: {entry.etymology}[/dim]")
    if entry.first_known_use:
        console.print(f"  [dim]First Known Use: {entry.first_known_use}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("words", nargs="+", help="One or more words to look up.")
    parser.add_argument("--no-fallback", action="store_true", help="Skip the Playwright site-scrape fallback; API only.")
    parser.add_argument("--headless", action="store_true", help="Run the fallback browser headless (more likely to get Cloudflare-challenged).")
    args = parser.parse_args()

    api_key = mw.mw_api_key()
    if not api_key:
        console.print("[yellow]No MW_DICTIONARY_API_KEY found (.env or env var) -- API tier disabled, scrape-only.[/yellow]")

    session = requests.Session()

    # The scrape fallback's browser is opened lazily (only if some word actually
    # needs it) and kept open for the whole run -- one persistent-context launch
    # for a batch of words, not one per word (which raced the profile-dir lock).
    scraper = None
    with ExitStack() as stack:
        for i, word in enumerate(args.words):
            if i:
                console.print()
            entries = mw.lookup_api(word, api_key, session, console=console) if api_key else []
            if not entries and not args.no_fallback:
                if scraper is None:
                    from concordance.mw_scrape import MWScraper
                    scraper = stack.enter_context(MWScraper(headless=args.headless))
                entries = scraper.lookup(word)
            if not entries:
                console.print(f"[bold]{word}[/bold] [red]— no entry found[/red] (API + scrape both came up empty)")
                continue
            for j, entry in enumerate(entries):
                if j:
                    console.print()
                _print_entry(entry)


if __name__ == "__main__":
    main()
