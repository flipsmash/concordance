"""Command-line entry point.

Three verbs:
  concordance run <book>            extract → filter → judge → enrich a book,
                                    writing <book>.vocab.csv (and a pristine copy
                                    to archive/) for you to hand-edit.
  concordance finalize <vocab.csv>  after you delete the rows you know / dislike,
                                    promote the survivors to master_vocab.csv and
                                    archive the book's files.
  concordance ingest <book>         same pipeline as `run`, but writes straight
                                    to Postgres (kept -> word/word_book, dropped
                                    -> rejected_word per book) — no CSV, no
                                    hand-edit, no finalize. Review/prune the
                                    result afterward in the review web app.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import Config
from .extract import ScannedPDFError, UnsupportedFormatError
from . import db
from .deepen import define as define_cmd
from .finalize import finalize_file
from .pipeline import process as process_pipeline
from .pipeline import run as run_pipeline

app = typer.Typer(add_completion=False, help="Extract interesting vocabulary from a book.")
console = Console()


@app.command()
def run(
    book: Path = typer.Argument(..., help="Path to an EPUB, text PDF, or .txt file."),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to a .gguf model. Defaults to the 14B; falls back to the stub judge if that file is absent."),
    stub: bool = typer.Option(False, "--stub", help="Force the no-model stub judge even if the default model is present."),
    min_zipf: float = typer.Option(3.5, "--min-zipf", help="Frequency floor; higher keeps rarer words only."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap the shortlist size (0 = no cap)."),
    no_lookup: bool = typer.Option(False, "--no-lookup", help="Skip online definition lookups."),
) -> None:
    """Run the extraction pipeline on a book."""
    cfg = Config(
        min_zipf=min_zipf,
        limit=limit,
        lookup_definitions=not no_lookup,
    )
    if stub:
        cfg.model_path = ""              # explicit opt-out of the model
    elif model:
        cfg.model_path = str(model)      # explicit override; else Config's 14B default
    try:
        result = run_pipeline(book, cfg, console)
    except (ScannedPDFError, UnsupportedFormatError, FileNotFoundError) as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1)

    console.print()
    console.rule("[bold green]Done[/bold green]")
    console.print(f"[bold]{len(result.kept)}[/bold] words → {result.vocab_path.name}")
    console.print(f"[dim]{len(result.rejected)} rejected → {result.rejected_path.name}[/dim]")


@app.command()
def ingest(
    book: Path = typer.Argument(..., help="Path to an EPUB, text PDF, or .txt file."),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to a .gguf model. Defaults to the 14B; falls back to the stub judge if that file is absent."),
    stub: bool = typer.Option(False, "--stub", help="Force the no-model stub judge even if the default model is present."),
    min_zipf: float = typer.Option(3.5, "--min-zipf", help="Frequency floor; higher keeps rarer words only."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap the shortlist size (0 = no cap)."),
    no_lookup: bool = typer.Option(False, "--no-lookup", help="Skip online definition lookups."),
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema to write into."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
    no_archive: bool = typer.Option(False, "--no-archive", help="Leave the source book file in place instead of moving it to archive/."),
) -> None:
    """Run the extraction pipeline and write straight to Postgres — no CSV,
    no hand-edit, no finalize. Review/prune the result in the review web app."""
    cfg = Config(
        min_zipf=min_zipf,
        limit=limit,
        lookup_definitions=not no_lookup,
    )
    if stub:
        cfg.model_path = ""              # explicit opt-out of the model
    elif model:
        cfg.model_path = str(model)      # explicit override; else Config's 14B default

    try:
        kept, rejected = process_pipeline(book, cfg, console)
    except (ScannedPDFError, UnsupportedFormatError, FileNotFoundError) as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1)

    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}")
        console.print("[dim]set DATABASE_URL in the environment or a .env file[/dim]")
        raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    stats = db.sync_book_results(conn, book.stem, kept, rejected, schema)
    conn.close()

    console.print()
    console.rule("[bold green]Done[/bold green]")
    console.print(f"[bold]{stats['kept']}[/bold] words kept → '{schema}'.word (+word_book)")
    console.print(f"[dim]{stats['rejected']} rejected → '{schema}'.rejected_word[/dim]")

    if not no_archive:
        archive_dir = book.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        dest = archive_dir / book.name
        try:
            shutil.move(str(book), str(dest))
            console.print(f"[dim]book moved → archive/{dest.name}[/dim]")
        except OSError as exc:
            console.print(f"[yellow]![/yellow] could not archive {book.name}: {exc}")


@app.command()
def define(
    vocab_csv: Path = typer.Argument(..., help="A <book>.vocab.csv; undefined rows are resolved / scored."),
    web: bool = typer.Option(False, "--web", help="Last resort: web-search + grounded LLM extraction for words no dictionary defines."),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Model for --web extraction (defaults to the 14B)."),
) -> None:
    """Resolve still-undefined words via deeper sources; score the rest for validity."""
    if not vocab_csv.exists():
        console.print(f"[red]✗[/red] no such file: {vocab_csv}")
        raise typer.Exit(code=1)
    define_cmd(vocab_csv, console, use_web=web, model_path=str(model) if model else None)


@app.command()
def finalize(
    vocab_csv: Path = typer.Argument(..., help="A hand-edited <book>.vocab.csv (every remaining row is added)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Add every remaining term in a hand-edited list to the master list, then archive the book."""
    if not vocab_csv.exists():
        console.print(f"[red]✗[/red] no such file: {vocab_csv}")
        raise typer.Exit(code=1)
    finalize_file(vocab_csv, console, assume_yes=yes)



@app.command("sync-db")
def sync_db(
    master_csv: Path = typer.Argument(Path("master_vocab.csv"), help="Master vocab CSV to load."),
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema to write into."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Sync master_vocab.csv into PostgreSQL (creates the schema, then upserts)."""
    if not master_csv.exists():
        console.print(f"[red]✗[/red] no such file: {master_csv}")
        raise typer.Exit(code=1)
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}")
        console.print("[dim]set DATABASE_URL in the environment or a .env file[/dim]")
        raise typer.Exit(code=1)
    trgm = db.apply_schema(conn, schema)
    stats = db.sync_master(master_csv, conn, schema)
    conn.close()
    console.print(
        f"[green]✓[/green] synced [bold]{stats['words']}[/bold] words, "
        f"{stats['books']} books, {stats['links']} word→book links into schema '{schema}'."
    )
    if not trgm:
        console.print("[dim]note: pg_trgm index skipped (needs CREATE EXTENSION privilege).[/dim]")




@app.command("load-taxonomy")
def load_taxonomy(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Create the category tables and load the USAS taxonomy into PostgreSQL."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}")
        raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    stats = db.load_taxonomy(conn, schema)
    conn.close()
    console.print(f"[green]✓[/green] loaded [bold]{stats['categories']}[/bold] USAS categories "
                  f"({stats['top_level']} top-level fields) into {schema}.category")



@app.command()
def classify(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Model (defaults to the 14B)."),
    limit: int = typer.Option(0, "--limit", "-l", help="Only classify the first N words (0 = all)."),
    only_missing: bool = typer.Option(False, "--only-missing", help="Only classify words that have no category yet."),
    batch: int = typer.Option(0, "--batch", help="Override the batch size (smaller = fewer omissions)."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Tag every word in the DB with USAS categories (LLM + WordNet-Domains prior)."""
    from .classify import classify_and_store
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    cfg = Config()
    if model:
        cfg.model_path = str(model)
    stats = classify_and_store(conn, schema, cfg, limit, only_missing=only_missing, batch=batch or None)
    conn.close()
    console.print(f"[green]✓[/green] classified [bold]{stats['classified']}[/bold]/{stats['words']} words "
                  f"-> {stats['assignments']} category assignments")



@app.command()
def archaic(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Set the archaic-currency flag (current/dated/archaic/obsolete) on word_difficulty."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    dist = db.compute_archaic(conn, schema)
    conn.close()
    total = sum(dist.values())
    parts = ", ".join(f"{k} {v}" for k, v in sorted(dist.items()))
    console.print(f"[green]✓[/green] archaic flags set on [bold]{total}[/bold] words — {parts}")



@app.command()
def ngram(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    refetch: bool = typer.Option(False, "--refetch", help="Refetch all words (default: only uncached)."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of words fetched."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Fetch + cache Google Books Ngram features (rarity + recency) into word_ngram."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    stats = db.fetch_ngrams(conn, schema, only_missing=not refetch, limit=limit)
    conn.close()
    console.print(f"[green]✓[/green] ngram: fetched [bold]{stats['fetched']}[/bold]/{stats['words']} "
                  f"({stats['in_corpus']} in corpus, {stats['failed']} failed)")



@app.command()
def difficulty(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Compute the ex-ante difficulty scalar (+ factor breakdown) on word_difficulty."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    stats = db.compute_difficulty(conn, schema)
    conn.close()
    console.print(f"[green]✓[/green] difficulty set on [bold]{stats['words']}[/bold] words "
                  f"(mean {stats['mean']}, median {stats['median']})")



@app.command()
def quizdef(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Model for rewrites (defaults to the 14B)."),
    refresh: bool = typer.Option(False, "--refresh", help="Recompute all (default: only words without a quiz definition)."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap words processed."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Build quiz-safe definitions: clean defs pass through, leaking ones are LLM-rewritten."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    cfg = Config()
    if model:
        cfg.model_path = str(model)
    stats = db.compute_quiz_definitions(conn, schema, cfg, only_missing=not refresh, limit=limit)
    conn.close()
    console.print(f"[green]✓[/green] quiz defs: [bold]{stats['words']}[/bold] words "
                  f"({stats['clean']} clean, {stats['rewritten']} rewritten, {stats['redacted']} redacted)")



@app.command()
def quizzable(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Flag words as quizzable (exclude grammatical/variant forms and trivially-inferable derivatives)."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    dist = db.compute_quizzable(conn, schema)
    conn.close()
    console.print(f"[green]✓[/green] quizzable: [bold]{dist.get('quizzable',0)}[/bold] quizzable, "
                  f"{dist.get('excluded',0)} excluded")


@app.command()
def commons_search(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    dump_path: str = typer.Option(None, "--dump-path", help="Path to the kaikki Wiktextract dump "
                                   "(default: data/wiktextract-en.jsonl.gz)."),
    refetch: bool = typer.Option(False, "--refetch", help="Re-check every word (default: only unchecked)."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of words searched."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Second-pass direct Commons search for real recordings kaikki's dump missed
    (confirmed to happen). Slow — Commons rate-limits hard; meant to run for hours.
    Stores only the search result; `audio` does the actual download."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    try:
        stats = db.search_commons_direct(conn, schema, dump_path=dump_path, only_missing=not refetch, limit=limit)
    except FileNotFoundError as exc:
        console.print(f"[red]✗[/red] {exc}"); raise typer.Exit(code=1)
    conn.close()
    console.print(f"[green]✓[/green] commons-search: [bold]{stats.get('total',0)}[/bold] candidates — "
                  f"{stats.get('found',0)} found, {stats.get('not_found',0)} not found, "
                  f"{stats.get('skipped_kaikki_has_audio',0)} skipped (kaikki already has audio)")


@app.command()
def wordnik_pron(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    refetch: bool = typer.Option(False, "--refetch", help="Re-check every word (default: only unchecked)."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of words checked."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Fetch RAW Wordnik pronunciations (ahd-5/arpabet/gcide-diacritical) into
    word.wordnik_pron_raw. No IPA conversion here — that's a separate `ipa` pass,
    so a converter bug never costs re-running this slow, rate-limited fetch."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    stats = db.fetch_wordnik_pronunciations(conn, schema, only_missing=not refetch, limit=limit)
    conn.close()
    if "error" in stats:
        console.print(f"[red]✗[/red] {stats['error']}"); raise typer.Exit(code=1)
    console.print(f"[green]✓[/green] wordnik-pron: [bold]{stats.get('words',0)}[/bold] checked — " +
                  ", ".join(f"{v} {k}" for k, v in stats.items() if k != "words"))


@app.command()
def ipa(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    dump_path: str = typer.Option(None, "--dump-path", help="Path to the kaikki Wiktextract dump "
                                   "(default: data/wiktextract-en.jsonl.gz)."),
    refetch: bool = typer.Option(False, "--refetch", help="Re-check every word (default: only empty/invalid ipa)."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of words checked."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Backfill + clean word.ipa from kaikki, then Wordnik (ARPAbet/AHD-5 converted,
    direct IPA as-is). NULLs out+replaces transcriptions that fail an
    English-language sanity check. Run this before `audio` — synthesis is only
    as good as the IPA it's given."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    try:
        stats = db.compute_ipa(conn, schema, dump_path=dump_path, only_missing=not refetch, limit=limit)
    except FileNotFoundError as exc:
        console.print(f"[red]✗[/red] {exc}"); raise typer.Exit(code=1)
    conn.close()
    backfilled = stats.get("backfilled_kaikki", 0) + stats.get("backfilled_wordnik", 0)
    corrected = stats.get("corrected_kaikki", 0) + stats.get("corrected_wordnik", 0)
    console.print(f"[green]✓[/green] ipa: {stats.get('total',0)} words — "
                  f"[bold]{stats.get('already_valid',0)}[/bold] already valid, "
                  f"{backfilled} backfilled ({stats.get('backfilled_kaikki',0)} kaikki, "
                  f"{stats.get('backfilled_wordnik',0)} wordnik), "
                  f"{corrected} corrected, "
                  f"{stats.get('cleared_no_replacement',0)} cleared (no valid source found), "
                  f"{stats.get('unresolved',0)} still unresolved")


@app.command()
def commons_download(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of files downloaded."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Download the real recordings `commons-search` confirmed exist, upgrading
    any word currently on Azure-synthesized or no-data to the real recording.
    Slow and deliberate — run separately from `audio`, which exhausted Commons'
    rate limit when interleaved with fast Azure calls."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    stats = db.download_commons_direct_finds(conn, schema, limit=limit)
    conn.close()
    console.print(f"[green]✓[/green] commons-download: [bold]{stats.get('downloaded',0)}[/bold]/"
                  f"{stats.get('candidates',0)} downloaded, {stats.get('failed',0)} failed "
                  f"(upgraded from azure: {stats.get('upgraded_from_azure',0)}, "
                  f"from none: {stats.get('upgraded_from_none',0)})")


@app.command()
def audio(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    dump_path: str = typer.Option(None, "--dump-path", help="Path to the kaikki Wiktextract dump "
                                   "(default: data/wiktextract-en.jsonl.gz)."),
    refetch: bool = typer.Option(False, "--refetch", help="Re-attempt all words (default: only ones with no word_audio row)."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of words processed."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Pronunciation audio: Commons recordings where they exist, else Azure IPA-guided
    synthesis where a transcription is known. Words with neither are left alone."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    try:
        stats = db.compute_audio(conn, schema, dump_path=dump_path, only_missing=not refetch, limit=limit)
    except FileNotFoundError as exc:
        console.print(f"[red]✗[/red] {exc}"); raise typer.Exit(code=1)
    conn.close()
    commons_total = stats.get('commons', 0) + stats.get('commons_direct_search', 0)
    console.print(f"[green]✓[/green] audio: [bold]{stats.get('candidates',0)}[/bold] processed — "
                  f"{commons_total} Commons ({stats.get('commons_direct_search',0)} via direct search), "
                  f"{stats.get('azure',0)} Azure-synthesized, "
                  f"{stats.get('none',0)} no data found")


@app.command()
def audio_guess(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of words synthesized."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Last resort for words with no real recording and no IPA anywhere: Azure
    guesses pronunciation from spelling alone. Recorded as source='azure_guess'
    — distinct from IPA-guided 'azure' — so the app can flag these unverified."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    stats = db.synthesize_unverified_guesses(conn, schema, limit=limit)
    conn.close()
    if "error" in stats:
        console.print(f"[red]✗[/red] {stats['error']}"); raise typer.Exit(code=1)
    console.print(f"[green]✓[/green] audio-guess: [bold]{stats.get('synthesized',0)}[/bold]/"
                  f"{stats.get('candidates',0)} synthesized (unverified), {stats.get('failed',0)} failed")


if __name__ == "__main__":
    app()
