"""Command-line entry point.

Three verbs:
  concordance run <book>            extract → filter → judge → enrich a book,
                                    writing <book>.vocab.csv (and a pristine copy
                                    to archive/) for you to hand-edit.
  concordance finalize <vocab.csv>  after you delete the rows you know / dislike,
                                    promote the survivors to master_vocab.csv and
                                    archive the book's files.
  concordance ingest [book]         same pipeline as `run`, but writes straight
                                    to Postgres (kept -> word/word_book, dropped
                                    -> rejected_word per book) — no CSV, no
                                    hand-edit, no finalize. Review/prune the
                                    result afterward in the review web app.
                                    Omit the argument to process every EPUB/
                                    PDF/txt file in incoming/ instead of a
                                    single named file.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import psycopg
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

INCOMING_DIR = Path("incoming")
ARCHIVE_DIR = Path("archive")
_INGEST_SUFFIXES = {".epub", ".pdf", ".txt"}


def _parse_incoming_name(path: Path) -> tuple[str, Optional[str]]:
    """'[Title] -- [Author].ext' -> (title, author). Falls back to the whole
    stem as title (author=None) if the ' -- ' delimiter isn't present, so an
    oddly-named file still ingests instead of erroring out."""
    stem = path.stem
    if " -- " in stem:
        title, author = stem.split(" -- ", 1)
        return title.strip(), author.strip() or None
    return stem.strip(), None


@app.command()
def run(
    book: Path = typer.Argument(..., help="Path to an EPUB, text PDF, or .txt file."),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to a .gguf model. Defaults to the 14B; falls back to the stub judge if that file is absent."),
    stub: bool = typer.Option(False, "--stub", help="Force the no-model stub judge even if the default model is present."),
    min_zipf: float = typer.Option(3.5, "--min-zipf", help="Frequency floor; higher keeps rarer words only."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap the shortlist size (0 = no cap)."),
    no_lookup: bool = typer.Option(False, "--no-lookup", help="Skip online definition lookups."),
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema to check for already-pruned words."),
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
        result = run_pipeline(book, cfg, console, schema=schema)
    except (ScannedPDFError, UnsupportedFormatError, FileNotFoundError) as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(code=1)
    except (RuntimeError, psycopg.Error) as exc:
        console.print(f"[red]✗[/red] cannot connect to Postgres: {exc}")
        console.print("[dim]the validity gate now needs vocab.wiktionary — "
                       "set DATABASE_URL in the environment or a .env file[/dim]")
        raise typer.Exit(code=1)

    console.print()
    console.rule("[bold green]Done[/bold green]")
    console.print(f"[bold]{len(result.kept)}[/bold] words → {result.vocab_path.name}")
    console.print(f"[dim]{len(result.rejected)} rejected → {result.rejected_path.name}[/dim]")


@app.command()
def ingest(
    book: Optional[Path] = typer.Argument(None, help="Path to an EPUB, text PDF, or .txt file. Omit to process every file in incoming/."),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to a .gguf model. Defaults to the 14B; falls back to the stub judge if that file is absent."),
    stub: bool = typer.Option(False, "--stub", help="Force the no-model stub judge even if the default model is present."),
    min_zipf: float = typer.Option(3.5, "--min-zipf", help="Frequency floor; higher keeps rarer words only."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap the shortlist size (0 = no cap)."),
    no_lookup: bool = typer.Option(False, "--no-lookup", help="Skip online definition lookups."),
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema to write into."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
    no_archive: bool = typer.Option(False, "--no-archive", help="Leave source book files in place instead of moving them to archive/."),
) -> None:
    """Run the extraction pipeline and write straight to Postgres — no CSV,
    no hand-edit, no finalize. Review/prune the result in the review web app.

    With no argument, processes every .epub/.pdf/.txt file in incoming/,
    parsing "[Title] -- [Author]" from each filename to set book.author."""
    cfg = Config(
        min_zipf=min_zipf,
        limit=limit,
        lookup_definitions=not no_lookup,
    )
    if stub:
        cfg.model_path = ""              # explicit opt-out of the model
    elif model:
        cfg.model_path = str(model)      # explicit override; else Config's 14B default

    if book is not None:
        books = [book]
        batch_mode = False
    else:
        if not INCOMING_DIR.is_dir():
            console.print(f"[red]✗[/red] no such directory: {INCOMING_DIR}/")
            raise typer.Exit(code=1)
        books = sorted(p for p in INCOMING_DIR.iterdir() if p.suffix.lower() in _INGEST_SUFFIXES)
        batch_mode = True
        if not books:
            console.print(f"[yellow]![/yellow] no .epub/.pdf/.txt files found in {INCOMING_DIR}/")
            raise typer.Exit(code=0)
        console.print(f"Found [bold]{len(books)}[/bold] file(s) in {INCOMING_DIR}/.")

    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}")
        console.print("[dim]set DATABASE_URL in the environment or a .env file[/dim]")
        raise typer.Exit(code=1)
    db.apply_schema(conn, schema)

    # Build the heavy, book-independent resources ONCE and reuse them across
    # every book — the ~9GB judge model (~49s to load), the spaCy model, and
    # the validity gate's SymSpell/WordNet/234k-word corpora. Reloading these
    # per book was pure per-book overhead (crippling on a large corpus).
    from . import judge as _judge, tokenize as _tokenize, validity as _validity
    with console.status("[bold]Loading model + resources…"):
        nlp = _tokenize.load_nlp()
        gate = _validity.ValidityGate(cfg)
        judge_obj = _judge.get_judge(cfg)

    for i, b in enumerate(books, 1):
        if batch_mode:
            console.print()
            console.rule(f"[bold]{i}/{len(books)} · {b.name}")
        title, author = _parse_incoming_name(b)

        try:
            kept, rejected = process_pipeline(b, cfg, console, schema=schema,
                                              nlp=nlp, gate=gate, judge_obj=judge_obj)
        except (ScannedPDFError, UnsupportedFormatError, FileNotFoundError, RuntimeError, psycopg.Error) as exc:
            console.print(f"[red]✗[/red] {exc}")
            if batch_mode:
                continue
            conn.close()
            raise typer.Exit(code=1)

        stats = db.sync_book_results(conn, title, kept, rejected, schema, author=author)
        console.print(
            f"[bold]{stats['kept']}[/bold] words kept, {stats['rejected']} rejected "
            f"→ '{schema}' (title={title!r}, author={author or 'unknown'})"
        )

        if not no_archive:
            # batch mode always archives to the top-level archive/ (incoming/
            # and archive/ are siblings at the project root); single-file mode
            # keeps `run`'s convention of archiving next to the source file.
            archive_dir = ARCHIVE_DIR if batch_mode else b.parent / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            dest = archive_dir / b.name
            try:
                shutil.move(str(b), str(dest))
                console.print(f"[dim]moved → {archive_dir}/{dest.name}[/dim]")
            except OSError as exc:
                console.print(f"[yellow]![/yellow] could not archive {b.name}: {exc}")

    conn.close()
    console.print()
    console.rule("[bold green]Done[/bold green]")


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
    try:
        define_cmd(vocab_csv, console, use_web=web, model_path=str(model) if model else None)
    except (RuntimeError, psycopg.Error) as exc:
        console.print(f"[red]✗[/red] cannot connect to Postgres: {exc}")
        console.print("[dim]the local dictionary lookup needs vocab.wiktionary — "
                       "set DATABASE_URL in the environment or a .env file[/dim]")
        raise typer.Exit(code=1)


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



@app.command("normalize-pos")
def normalize_pos_cmd(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Clean up word.part_of_speech: fold abbreviations/case variants (adj,
    adv, pron, propn, x, Noun, ...) down to one consistent, spelled-out
    vocabulary. Idempotent — safe to re-run any time."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    stats = db.normalize_word_pos(conn, schema)
    conn.close()
    console.print(f"[green]✓[/green] normalize-pos: [bold]{stats['changed']}[/bold]/{stats['words']} words updated")


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
def refill(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap words processed (0 = all)."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Backfill blank definitions for already-accepted words (local Wiktionary,
    then Free Dictionary API / online Wiktionary — same cheap sources `ingest`
    tries). Words that stay undefined keep their permanent `flagged_undefined`
    marker regardless; this only ever fills the definition, it never clears it."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    with console.status("[bold]Backfilling definitions…"):
        stats = db.refill_definitions(conn, schema, limit=limit)
    conn.close()
    console.print(f"[green]✓[/green] refill: [bold]{stats['filled']}[/bold]/{stats['attempted']} "
                  f"filled ({stats['still_missing']} still undefined)")


@app.command()
def deepen(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    web: bool = typer.Option(False, "--web", help="Last resort: web-search + grounded LLM extraction."),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Model for --web extraction (defaults to the 14B)."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap words processed (0 = all)."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Run after `refill`: reach further (Wordnik, yourdictionary, optionally
    --web) for words still undefined. Whatever STILL can't be defined gets a
    validity estimate (real word vs OCR/variant/foreign/nonsense) written to
    word.validity_* — pair with the flagged_undefined marker to find prune
    candidates: flagged AND validity_label='likely-artifact' is the review
    queue. Needs WORDNIK_API_KEY in .env for the Wordnik source; falls back to
    yourdictionary-only without it."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    with console.status("[bold]Resolving the undefined tail…"):
        stats = db.deepen_definitions(conn, schema, use_web=web,
                                      model_path=str(model) if model else None, limit=limit)
    conn.close()
    console.print(f"[green]✓[/green] deepen: [bold]{stats['defined']}[/bold]/{stats['attempted']} "
                  f"defined ({stats['still_undefined']} scored for validity, still undefined)")


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
