#!/usr/bin/env python3
"""Stand-alone CLI: look up a term's definition from the web.

No database connection needed -- unlike `concordance define` (which resolves
an entire book's undefined words and needs Postgres for the local Wiktionary
dump), this is a single-word, network-only lookup you can run from anywhere.

Tries a quality-ordered cascade of sources, stopping at the first real hit:

  1. Wordnik (Century / GCIDE / AHD)  -- the archaic/literary vocabulary this
                                          project cares about most; needs
                                          WORDNIK_API_KEY (.env or env var),
                                          skipped silently if absent.
  2. Free Dictionary API              -- a real, curated modern dictionary.
  3. Wiktionary (REST)                -- broad community coverage.
  4. yourdictionary.com               -- a keyless scraped aggregator; lower
                                          confidence, but sometimes has nonce
                                          words the above miss.
  5. Web search + local LLM extraction -- last resort, ALWAYS tried if
                                          nothing above defined the word (no
                                          flag to disable). The model reads
                                          real search-result snippets and
                                          extracts a definition that is
                                          actually present in them -- it
                                          never invents one. The model is
                                          loaded lazily, only if this stage
                                          is actually reached, and skipped
                                          with a clear message if the model
                                          file isn't present.

Usage:
    python scripts/lookup_word.py cangue
    python scripts/lookup_word.py cangue armiger silkman
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from concordance import deepdef, dictionary, websearch  # noqa: E402
from concordance.model import Candidate  # noqa: E402

console = Console()


def _load_llm():
    """Lazy, best-effort local-model load for the web-search fallback. None
    (with a printed reason) if the model file isn't present -- this stage is
    genuinely optional at the infrastructure level even though it's always
    attempted when reached, per the "no flag to disable it" design."""
    from concordance.config import Config

    cfg = Config()
    if not cfg.model_path or not Path(cfg.model_path).exists():
        console.print(f"[dim]  (web-search/LLM fallback unavailable -- model not found at {cfg.model_path!r})[/dim]")
        return None
    from llama_cpp import Llama

    console.print("[dim]  loading local model for web-search fallback…[/dim]")
    return Llama(model_path=cfg.model_path, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)


class _LazyLLM:
    """Loads the model at most once, only if a lookup actually reaches the
    web-search stage -- most words resolve via a real dictionary long before
    that, and shouldn't pay the model-load cost."""

    def __init__(self):
        self._llm = None
        self._tried = False

    def get(self):
        if not self._tried:
            self._tried = True
            self._llm = _load_llm()
        return self._llm


def lookup(word: str, session, lazy_llm: _LazyLLM) -> Candidate | None:
    """The quality-ordered cascade described in the module docstring. Returns
    a Candidate with definition/source (+ whatever else that source carries:
    POS, IPA, etymology, synonyms) on a hit, or None if every source --
    including the web-search fallback -- came up empty."""
    cand = Candidate(lemma=word, pos="")

    key = deepdef.wordnik_key()
    if key and deepdef._from_wordnik(cand, session, key):
        return cand

    dictionary.enrich(cand, session)  # tries Free Dictionary, then Wiktionary
    if cand.definition:
        return cand

    if deepdef._from_yourdictionary(cand, session):
        return cand

    llm = lazy_llm.get()
    if llm is not None and websearch.define_via_web(cand, llm):
        return cand

    return None


def _print_result(word: str, cand: Candidate | None) -> None:
    if cand is None:
        console.print(f"[bold]{word}[/bold] [red]— no definition found[/red] (tried every source)")
        return
    console.print(f"[bold]{word}[/bold] [dim]({cand.definition_source})[/dim]")
    pos_str = f"[italic]{cand.part_of_speech}[/italic]  " if cand.part_of_speech else ""
    console.print(f"  {pos_str}{cand.definition}")
    if cand.ipa:
        console.print(f"  [dim]IPA: {cand.ipa}[/dim]")
    if cand.etymology:
        console.print(f"  [dim]Etymology: {cand.etymology}[/dim]")
    if cand.synonyms:
        console.print(f"  [dim]Synonyms: {', '.join(cand.synonyms)}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("words", nargs="+", help="One or more terms to look up.")
    parser.add_argument(
        "--delay", type=float, default=13.0,
        help="Seconds to wait between words (default 13.0). The cascade tries Wordnik FIRST "
             "for every word, whether or not it hits -- a free-tier Wordnik key is capped at "
             "5 requests/minute, so anything faster than ~12s/word will start drawing 429s and "
             "wasting time in retry-backoff rather than actually going faster. Irrelevant for a "
             "single word; matters once you're looking up more than a handful in one run.",
    )
    args = parser.parse_args()

    session = dictionary.make_session()
    lazy_llm = _LazyLLM()

    for i, word in enumerate(args.words):
        if i:
            console.print()
            time.sleep(args.delay)
        cand = lookup(word, session, lazy_llm)
        _print_result(word, cand)


if __name__ == "__main__":
    main()
