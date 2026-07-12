"""Orchestration — run the whole pipeline (§03) on one book."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from . import clean, dictionary, extract, floor, judge, master, output, propernouns, tokenize, validity
from .config import Config
from .model import Candidate, Verdict


@dataclass
class Result:
    kept: list[Candidate]
    rejected: list[Candidate]
    vocab_path: Path
    rejected_path: Path


def process(book: str | Path, cfg: Config, console: Console | None = None) -> tuple[list[Candidate], list[Candidate]]:
    """Extract -> filter -> judge -> enrich a book, returning (kept, rejected).
    Shared by `run` (writes CSVs for hand-editing) and `ingest` (writes straight
    to Postgres) — everything through enrichment is identical; only what
    happens with the result differs."""
    console = console or Console()
    book = Path(book)

    with console.status("[bold]Extracting text…"):
        chapters = extract.extract(book)
        for ch in chapters:
            ch.text = clean.clean(ch.text)
    console.print(f"Extracted [bold]{len(chapters)}[/bold] section(s) from {book.name}.")

    with console.status("[bold]Tokenizing & lemmatizing…"):
        candidates = tokenize.tokenize(chapters)
    console.print(f"Found [bold]{len(candidates)}[/bold] distinct lemmas.")

    # --- deterministic filters -------------------------------------------
    floor.apply_floor(candidates, cfg)
    propernouns.strip_proper_nouns(candidates, cfg)
    with console.status("[bold]Checking validity…"):
        validity.apply_validity(candidates, cfg)
    survivors = [c for c in candidates.values() if c.verdict in (Verdict.KEEP, Verdict.UNSURE)]
    console.print(f"[bold]{len(survivors)}[/bold] candidates survived the floor + validity gate.")

    # --- LLM interestingness judge ---------------------------------------
    with console.status("[bold]Judging interestingness…"):
        judge.get_judge(cfg).judge(list(candidates.values()))
    shortlist = [c for c in candidates.values() if c.verdict in (Verdict.KEEP, Verdict.UNSURE)]
    shortlist.sort(key=lambda c: (c.zipf, c.lemma))
    if cfg.limit:
        for extra in shortlist[cfg.limit:]:
            extra.verdict = Verdict.DROP
        shortlist = shortlist[: cfg.limit]
    console.print(f"[bold]{len(shortlist)}[/bold] words on the shortlist.")

    # --- enrichment ------------------------------------------------------
    if cfg.lookup_definitions and shortlist:
        session = dictionary.make_session()
        with console.status("[bold]Looking up definitions…") as status:
            for i, cand in enumerate(shortlist, 1):
                dictionary.enrich(cand, session)
                status.update(f"[bold]Looking up definitions… {i}/{len(shortlist)}")

    return output.partition(candidates)


def run(book: str | Path, cfg: Config, console: Console | None = None) -> Result:
    console = console or Console()
    book = Path(book)
    kept, rejected = process(book, cfg, console)

    # --- write + snapshot ------------------------------------------------
    # No interactive pass: the shortlist is written whole for the user to hand-edit
    # (delete rows they know / dislike), then `concordance finalize` promotes the
    # survivors. A pristine copy is archived immediately so the original and the
    # cleaned version both persist.
    stem = book.with_suffix("")
    vocab_path = Path(f"{stem}.vocab.csv")
    rejected_path = Path(f"{stem}.rejected.csv")
    output.write_vocab(vocab_path, kept)
    output.write_rejected(rejected_path, rejected)

    snapshot = master.snapshot_original(vocab_path, book.parent / "archive")
    console.print(f"[dim]pristine copy → archive/{snapshot.name}[/dim]")

    return Result(kept=kept, rejected=rejected, vocab_path=vocab_path, rejected_path=rejected_path)
