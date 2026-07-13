"""Orchestration — run the whole pipeline (§03) on one book."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from . import clean, db, dictionary, extract, floor, judge, localdict, master, output, propernouns, tokenize, validity
from .config import Config
from .model import Candidate, RejectReason, Verdict


@dataclass
class Result:
    kept: list[Candidate]
    rejected: list[Candidate]
    vocab_path: Path
    rejected_path: Path


def apply_pruned_exclusions(candidates: dict[str, Candidate], pruned: set[str]) -> int:
    """Mark every candidate matching a lemma a human has already manually
    pruned (word.active=false in a previous book) as DROP/ALREADY_KNOWN,
    before any other stage gets a chance to (re-)decide it. Returns the count
    excluded. Pure — no DB access — so it's unit-testable on its own."""
    matches = [c for c in candidates.values() if c.verdict is None and c.lemma in pruned]
    for c in matches:
        c.verdict = Verdict.DROP
        c.reject_reason = RejectReason.ALREADY_KNOWN
    return len(matches)


def process(book: str | Path, cfg: Config, console: Console | None = None,
            schema: str = db.DEFAULT_SCHEMA) -> tuple[list[Candidate], list[Candidate]]:
    """Extract -> filter -> judge -> enrich a book, returning (kept, rejected).
    Shared by `run` (writes CSVs for hand-editing) and `ingest` (writes straight
    to Postgres) — everything through enrichment is identical; only what
    happens with the result differs.

    Requires a live DATABASE_URL for the local Wiktionary dump (vocab.wiktionary,
    ~500k terms) — checked first in the validity gate and tried first during
    enrichment, both because it's free (no network) and because unlike every
    other authority here it carries no "Proper noun" POS at all, so it doesn't
    get to vouch for real names the way the frequency-based checks do."""
    console = console or Console()
    book = Path(book)
    conn = db.connect()

    with console.status("[bold]Extracting text…"):
        chapters = extract.extract(book)
        for ch in chapters:
            ch.text = clean.clean(ch.text)
    console.print(f"Extracted [bold]{len(chapters)}[/bold] section(s) from {book.name}.")

    with console.status("[bold]Tokenizing & lemmatizing…"):
        candidates = tokenize.tokenize(chapters)
    console.print(f"Found [bold]{len(candidates)}[/bold] distinct lemmas.")

    lexicon = localdict.build_lexicon(conn, set(candidates.keys()))

    # Checked before anything else, even the frequency floor: a word a human
    # has already manually pruned (word.active=false) as too common/easy
    # shouldn't get re-decided — and re-judged, at real LLM cost — from
    # scratch every time it turns up in a new book.
    pruned = db.fetch_pruned_lemmas(conn, schema)
    n_excluded = apply_pruned_exclusions(candidates, pruned)
    if n_excluded:
        console.print(f"[dim]{n_excluded} already manually pruned in a previous book — skipped.[/dim]")

    # --- deterministic filters -------------------------------------------
    floor.apply_floor(candidates, cfg)
    propernouns.strip_proper_nouns(candidates, cfg)
    with console.status("[bold]Checking validity…"):
        validity.apply_validity(candidates, cfg, local_dict=lexicon)
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
                if not localdict.enrich(cand, lexicon):
                    dictionary.enrich(cand, session)
                status.update(f"[bold]Looking up definitions… {i}/{len(shortlist)}")

    conn.close()
    return output.partition(candidates)


def run(book: str | Path, cfg: Config, console: Console | None = None,
        schema: str = db.DEFAULT_SCHEMA) -> Result:
    console = console or Console()
    book = Path(book)
    kept, rejected = process(book, cfg, console, schema=schema)

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
