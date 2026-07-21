#!/usr/bin/env python3
"""Stand-alone CLI: look up a term's definition from the web.

No database connection needed -- unlike `concordance define` (which resolves
an entire book's undefined words and needs Postgres for the local Wiktionary
dump), this is a single-word, network-only lookup you can run from anywhere.

Tries concordance.resolve's shared cascade (see resolve.py's own docstring
for the full rationale), stopping at the first real hit:

  1. Free Dictionary API / Wiktionary (REST) -- no key, no rate limit.
  2. Wordnik (Century / GCIDE / AHD)  -- the archaic/literary vocabulary this
                                          project cares about most; needs
                                          WORDNIK_API_KEY (.env or env var),
                                          skipped silently if absent. Tried
                                          AFTER the free tier now (this
                                          script used to try it first) to
                                          protect its tight 5-req/min budget
                                          for words nothing free can resolve.
  3. yourdictionary.com               -- a keyless scraped aggregator; lower
                                          confidence, but sometimes has nonce
                                          words the above miss.
  4. Web search + local LLM extraction -- last resort, ALWAYS tried if
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

from concordance import deepdef, dictionary, resolve  # noqa: E402
from concordance.model import Candidate  # noqa: E402

# soft_wrap=True: never insert a hard line break mid-sentence. Rich defaults
# to wrapping at ~80 columns even when stdout isn't a real terminal (e.g.
# redirected to a file for a batch run), which silently truncated wrapped
# continuation lines when a downstream parser only understood single-line
# entries -- real data loss, not just a cosmetic wrap.
console = Console(soft_wrap=True)


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
    """Stands in for a real Llama instance as resolve_definition's `llm`
    argument, so it's always "available" from the cascade's point of view
    (never None) but only actually loads the model the first time
    create_chat_completion is called -- i.e. only once the WEB tier is
    truly reached. Most words resolve via a real dictionary long before
    that, and shouldn't pay the model-load cost. If the model file isn't
    present, degrades to answering NONE (websearch.extract_definition's own
    "nothing found" signal) instead of crashing -- same as the old
    lazy_llm.get() returning None used to make the web tier a silent no-op."""

    def __init__(self):
        self._llm = None
        self._tried = False

    def create_chat_completion(self, *args, **kwargs):
        if not self._tried:
            self._tried = True
            self._llm = _load_llm()
        if self._llm is None:
            return {"choices": [{"message": {"content": "NONE"}}]}
        return self._llm.create_chat_completion(*args, **kwargs)


def lookup(word: str, session, lazy_llm: _LazyLLM) -> Candidate | None:
    """The shared cascade (concordance.resolve), stopping at the first real
    hit. `lazy_llm` is passed as the WEB tier's `llm` -- it satisfies
    websearch.define_via_web's `llm.create_chat_completion(...)` interface
    but only actually loads the model the first time that's called, i.e.
    only if every earlier tier missed AND a web search actually returned
    snippets worth asking the model about (extract_definition itself skips
    the model call when there are no snippets). Returns a Candidate with
    definition/source (+ whatever else that source carries: POS, IPA,
    etymology, synonyms) on a hit, or None if every source came up empty."""
    cand = Candidate(lemma=word, pos="")
    resolve.resolve_definition(
        cand, max_tier=resolve.Tier.WEB, session=session,
        wordnik_key=deepdef.wordnik_key(), llm=lazy_llm)
    return cand if cand.definition else None


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
        "--delay", type=float, default=0.5,
        help="Seconds to wait between words (default 0.5). Wordnik's 5-req/min free-tier cap "
             "is now paced internally by concordance.resolve, only when a word actually reaches "
             "that tier -- this flag is just polite self-throttling between words for the free/"
             "yourdictionary/web-search tiers, not a Wordnik budget. Irrelevant for a single word.",
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
