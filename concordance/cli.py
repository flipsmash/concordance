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
        cast_out_note = f", {stats['cast_out']} already-active word(s) cast out" if stats.get("cast_out") else ""
        console.print(
            f"[bold]{stats['kept']}[/bold] words kept, {stats['rejected']} rejected"
            f"{cast_out_note} → '{schema}' (title={title!r}, author={author or 'unknown'})"
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
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of words processed."),
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
    stats = db.normalize_word_pos(conn, schema, limit=limit)
    conn.close()
    console.print(f"[green]✓[/green] normalize-pos: [bold]{stats['changed']}[/bold]/{stats['words']} words updated")


@app.command()
def archaic(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of words processed."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Set the archaic-currency flag (current/dated/archaic/obsolete) on word_difficulty."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    dist = db.compute_archaic(conn, schema, limit=limit)
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
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of words processed."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Compute the ex-ante difficulty scalar (+ factor breakdown) on word_difficulty."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    stats = db.compute_difficulty(conn, schema, limit=limit)
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
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of words processed."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Flag words as quizzable (exclude grammatical/variant forms and trivially-inferable derivatives)."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    dist = db.compute_quizzable(conn, schema, limit=limit)
    conn.close()
    console.print(f"[green]✓[/green] quizzable: [bold]{dist.get('quizzable',0)}[/bold] quizzable, "
                  f"{dist.get('excluded',0)} excluded")


@app.command("book-similarity")
def book_similarity(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    top_k: int = typer.Option(12, "--top-k", help="Related books stored per book."),
    min_shared_words: int = typer.Option(3, "--min-shared-words",
                                          help="Minimum shared rare-word count for a pair to be stored at all."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of books (re)computed."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Compute each book's top-k most vocabulary-related books (IDF-weighted
    cosine similarity over shared active words -- lexical usage overlap,
    not semantic similarity; a different axis from the word-embedding
    graph). Always recomputes every book in scope -- IDF weights are
    corpus-wide and shift whenever any book's vocabulary changes."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    with console.status("[bold]Computing book vocabulary overlap…"):
        stats = db.compute_book_similarity(conn, schema, limit=limit, top_k=top_k,
                                           min_shared_words=min_shared_words)
    conn.close()
    console.print(f"[green]✓[/green] book-similarity: [bold]{stats['books']}[/bold] books "
                  f"-> [bold]{stats['pairs_stored']}[/bold] related-book pairs stored")


@app.command("author-similarity")
def author_similarity(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    top_k: int = typer.Option(12, "--top-k", help="Related authors stored per author."),
    min_shared_words: int = typer.Option(3, "--min-shared-words",
                                          help="Minimum shared rare-word count for a pair to be stored at all."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of authors (re)computed."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Compute each author's top-k most vocabulary-related authors (same
    IDF-weighted cosine metric as book-similarity, one level up -- an
    author's vector is the union of their books' word sets). Originally
    computed on demand per-request; moved to a precomputed table once real
    corpus scale (~3,500 authors) made the on-demand query take ~39s.
    Always recomputes every author in scope, same reasoning as
    book-similarity."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    with console.status("[bold]Computing author vocabulary overlap…"):
        stats = db.compute_author_similarity(conn, schema, limit=limit, top_k=top_k,
                                             min_shared_words=min_shared_words)
    conn.close()
    console.print(f"[green]✓[/green] author-similarity: [bold]{stats['authors']}[/bold] authors "
                  f"-> [bold]{stats['pairs_stored']}[/bold] related-author pairs stored")


@app.command("dedupe-plurals")
def dedupe_plurals(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    web: bool = typer.Option(True, "--web/--no-web",
                              help="Resolve a newly-created singular's definition through the full "
                                   "cascade including web-search + grounded local-LLM extraction. "
                                   "On by default, matching `deepen`; --no-web stops short of loading "
                                   "a model (faster, no GPU use)."),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Model for --web extraction (defaults to the 14B)."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap words processed (0 = all)."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Consolidate redundant plural-form entries ("warrs" -> "plural of warr")
    into their singular. A "plural of X" definition isn't real vocabulary
    content on its own -- quizdef.quizzable() already excludes these from
    quizzes -- so this resolves/creates the singular X (via the same cascade
    every other definition path uses) and soft-deletes the plural
    (active=false, reversible via the review webapp, never a hard delete).
    A singular that already exists but is currently inactive is always left
    untouched -- that's very likely a deliberate prior decision (human or
    automated), not something a plural merely existing should override."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    with console.status("[bold]Consolidating plural-form definitions…"):
        stats = db.dedupe_plural_definitions(conn, schema, limit=limit, use_web=web,
                                             model_path=str(model) if model else None)
    conn.close()
    console.print(f"[green]✓[/green] dedupe-plurals: [bold]{stats['attempted']}[/bold] plural definitions "
                  f"examined ({stats['unparsed']} not a clean 'plural of X' pattern) — "
                  f"[bold]{stats['linked']}[/bold] linked to an existing singular, "
                  f"[bold]{stats['created']}[/bold] singulars created "
                  f"({stats['still_undefined']} still undefined, {stats['cast_out']} cast out), "
                  f"{stats['left_inactive']} left inactive (deliberate prior decision)")


@app.command("expand-synonyms")
def expand_synonyms(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    web: bool = typer.Option(True, "--web/--no-web",
                              help="Resolve a synonym target's definition through the full cascade "
                                   "including web-search + grounded local-LLM extraction. On by "
                                   "default, matching `deepen`/`dedupe-plurals`; --no-web stops short "
                                   "of loading a model (faster, no GPU use)."),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Model for --web extraction (defaults to the 14B)."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap words processed (0 = all)."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Fix a real data-quality gap, not just a quizzability one: a definition
    that just says "synonym of X" ("ephebus" -> "Synonym of ephebe.") isn't
    real content, and unlike "plural of X" it isn't even excluded from
    quizzing today. Unlike dedupe-plurals, this never deletes/deactivates
    the word -- a synonym is a genuinely distinct headword worth keeping on
    its own, not redundant scaffolding for the same word. Instead it
    replaces the cross-reference definition with real content (extracted
    directly when the source already embedded a gloss, otherwise reused or
    freshly resolved from X, which gets created as its own word if it
    doesn't exist yet -- same cascade every other definition path uses). A
    target that exists but is currently inactive is always left untouched,
    and never used as a source to "upgrade" another word's definition --
    likely a deliberate earlier decision a synonym pointer isn't good reason
    to override."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    with console.status("[bold]Expanding synonym-only definitions…"):
        stats = db.expand_synonym_definitions(conn, schema, limit=limit, use_web=web,
                                              model_path=str(model) if model else None)
    conn.close()
    console.print(f"[green]✓[/green] expand-synonyms: [bold]{stats['attempted']}[/bold] synonym definitions "
                  f"examined ({stats['unparsed']} not a clean 'synonym of X' pattern) — "
                  f"[bold]{stats['extracted']}[/bold] had an embedded gloss extracted directly, "
                  f"[bold]{stats['reused_existing']}[/bold] reused an existing target's definition, "
                  f"[bold]{stats['target_created']}[/bold] targets newly resolved and created "
                  f"({stats['target_still_undefined']} still undefined, {stats['target_cast_out']} cast out), "
                  f"{stats['target_inactive']} left unchanged (target exists but inactive)")


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
                  f"filled ({stats['still_missing']} still undefined, "
                  f"{stats.get('cast_out', 0)} cast out as symbol/proper-noun-only)")


@app.command()
def deepen(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    web: bool = typer.Option(True, "--web/--no-web",
                              help="Web-search + grounded LLM extraction as the last resort. On by "
                                   "default -- it's where most of a deepen run's real yield comes "
                                   "from; pass --no-web to skip it (faster, no model load, but "
                                   "misses most of the permanently-undefined tail)."),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Model for --web extraction (defaults to the 14B)."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap words processed (0 = all)."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Run after `refill`: reach further (Wordnik, yourdictionary, web-search
    by default) for words still undefined. Whatever STILL can't be defined
    gets a validity estimate (real word vs OCR/variant/foreign/nonsense)
    written to word.validity_* — pair with the flagged_undefined marker to
    find prune candidates: flagged AND validity_label='likely-artifact' is
    the review queue. Needs WORDNIK_API_KEY in .env for the Wordnik source;
    falls back to yourdictionary+web without it."""
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
                  f"defined ({stats['still_undefined']} scored for validity, still undefined, "
                  f"{stats.get('cast_out', 0)} cast out as symbol/proper-noun-only)")


_FASTTEXT_MODEL_PATH = Path("models/fasttext_corpus.bin")


@app.command("train-fasttext")
def train_fasttext_cmd(
    archive_dir: Path = typer.Option(Path("archive"), "--archive-dir", help="Directory of already-ingested book files to train on."),
    model_path: Path = typer.Option(_FASTTEXT_MODEL_PATH, "--model-path", help="Where to write the trained model."),
    refresh: bool = typer.Option(False, "--refresh", help="Retrain even if a model already exists at model-path."),
) -> None:
    """(Re)train the FastText subword model on every archived book's text —
    a holistic pass over the whole corpus, not incremental, so this is run
    occasionally (e.g. after a large ingest batch), not per-book. Powers
    `concordance embed --signal fasttext`, which needs this model to exist."""
    from . import embed as _embed

    if model_path.exists() and not refresh:
        console.print(f"[yellow]![/yellow] {model_path} already exists — use --refresh to retrain.")
        raise typer.Exit(code=0)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path = model_path.with_suffix(".corpus.txt")
    with console.status("[bold]Building training corpus from archive/…"):
        n_files = _embed.build_fasttext_corpus(archive_dir, corpus_path)
    console.print(f"[dim]corpus built from {n_files} archived file(s) → {corpus_path.name}[/dim]")
    with console.status("[bold]Training FastText model…"):
        _embed.train_fasttext(corpus_path, model_path)
    console.print(f"[green]✓[/green] trained → {model_path}")


@app.command()
def embed(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    signal: str = typer.Option("definition", "--signal", help="'definition', 'fasttext', or 'both'."),
    fasttext_model: Path = typer.Option(_FASTTEXT_MODEL_PATH, "--fasttext-model", help="Trained model from train-fasttext."),
    refresh: bool = typer.Option(False, "--refresh", help="Recompute all (default: only words missing a vector)."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap words processed (0 = all)."),
    batch: int = typer.Option(64, "--batch", help="Definition-embedding batch size."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Compute per-word semantic-distance vectors into word_embedding: a
    sentence-embedding of each word's definition (meaning), and/or a FastText
    subword vector of the word form itself (works even with no definition —
    run `train-fasttext` first). Neither is an all-pairs distance matrix —
    both are queried on demand via a pgvector HNSW index, so this scales as
    the corpus grows instead of recomputing everything each time."""
    if signal not in ("definition", "fasttext", "both"):
        console.print("[red]✗[/red] --signal must be 'definition', 'fasttext', or 'both'"); raise typer.Exit(code=1)
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)

    if signal in ("definition", "both"):
        with console.status("[bold]Embedding definitions…"):
            stats = db.compute_definition_embeddings(conn, schema, only_missing=not refresh,
                                                      limit=limit, batch=batch)
        console.print(f"[green]✓[/green] definition: [bold]{stats['embedded']}[/bold]/{stats['words']} "
                      f"embedded ({stats['skipped_no_text']} skipped, no text)")
    if signal in ("fasttext", "both"):
        if not fasttext_model.exists():
            console.print(f"[red]✗[/red] {fasttext_model} not found — run `concordance train-fasttext` first.")
            raise typer.Exit(code=1)
        with console.status("[bold]Embedding word forms (FastText)…"):
            stats = db.compute_fasttext_embeddings(conn, schema, model_path=str(fasttext_model),
                                                   only_missing=not refresh, limit=limit)
        console.print(f"[green]✓[/green] fasttext: [bold]{stats['embedded']}[/bold]/{stats['words']} embedded")
    conn.close()


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
    backfilled = (stats.get("backfilled_kaikki", 0) + stats.get("backfilled_wordnik", 0)
                  + stats.get("backfilled_local_wiktionary", 0))
    corrected = (stats.get("corrected_kaikki", 0) + stats.get("corrected_wordnik", 0)
                 + stats.get("corrected_local_wiktionary", 0))
    console.print(f"[green]✓[/green] ipa: {stats.get('total',0)} words — "
                  f"[bold]{stats.get('already_valid',0)}[/bold] already valid, "
                  f"{backfilled} backfilled ({stats.get('backfilled_kaikki',0)} kaikki, "
                  f"{stats.get('backfilled_wordnik',0)} wordnik, "
                  f"{stats.get('backfilled_local_wiktionary',0)} local wiktionary), "
                  f"{corrected} corrected, "
                  f"{stats.get('cleared_no_replacement',0)} cleared (no valid source found), "
                  f"{stats.get('unresolved',0)} still unresolved")


@app.command()
def maintain(
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    model: Optional[Path] = typer.Option(None, "--model", "-m",
                                          help="Model for classify/quizdef/deepen --web (defaults to the 14B)."),
    dump_path: str = typer.Option(None, "--dump-path", help="Path to the kaikki Wiktextract dump "
                                   "(default: data/wiktextract-en.jsonl.gz)."),
    fasttext_model: Path = typer.Option(_FASTTEXT_MODEL_PATH, "--fasttext-model",
                                         help="Trained model from train-fasttext, for the embed step."),
    deepen_web: bool = typer.Option(True, "--deepen-web/--no-deepen-web",
                                     help="Let fill-definitions fall back to web-search + LLM extraction "
                                          "as the true last resort. On by default -- real-scale testing "
                                          "found it's where nearly all of a deepen pass's yield actually "
                                          "comes from (every other tier had already been tried by the "
                                          "time a word reaches it). Loads a local 14B model and is far "
                                          "slower per word than the other tiers; --no-deepen-web skips it."),
    recheck_after_days: int = typer.Option(14, "--recheck-after-days",
                                            help="Skip a word in fill-definitions if its last validity check "
                                                 "(i.e. its last failed resolution attempt) was more recent "
                                                 "than this many days ago -- without this, every maintain run "
                                                 "re-grinds the whole permanently-undefined tail through "
                                                 "Wordnik/web-search again, forever."),
    limit: int = typer.Option(0, "--limit", "-l", help="Cap number of words processed, per step (0 = all)."),
    skip_fill_definitions: bool = typer.Option(False, "--skip-fill-definitions",
                                                help="Skip the definition-lookup step (formerly two separate "
                                                     "refill + deepen steps, now one)."),
    skip_refill: bool = typer.Option(False, "--skip-refill",
                                      help="Deprecated alias for --skip-fill-definitions (refill and deepen "
                                           "are now one step; either flag skips it)."),
    skip_deepen: bool = typer.Option(False, "--skip-deepen",
                                      help="Deprecated alias for --skip-fill-definitions (refill and deepen "
                                           "are now one step; either flag skips it)."),
    skip_classify: bool = typer.Option(False, "--skip-classify"),
    skip_normalize_pos: bool = typer.Option(False, "--skip-normalize-pos"),
    skip_ngram: bool = typer.Option(False, "--skip-ngram"),
    skip_archaic: bool = typer.Option(False, "--skip-archaic"),
    skip_difficulty: bool = typer.Option(False, "--skip-difficulty"),
    skip_quizdef: bool = typer.Option(False, "--skip-quizdef"),
    skip_quizzable: bool = typer.Option(False, "--skip-quizzable"),
    skip_book_similarity: bool = typer.Option(False, "--skip-book-similarity"),
    skip_author_similarity: bool = typer.Option(False, "--skip-author-similarity"),
    skip_wordnik: bool = typer.Option(False, "--skip-wordnik", help="Skip the wordnik-pron fetch step."),
    skip_ipa: bool = typer.Option(False, "--skip-ipa", help="Skip the ipa backfill step."),
    skip_embed: bool = typer.Option(False, "--skip-embed"),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """Run the full post-ingest maintenance chain in dependency order:
    fill-definitions -> classify -> normalize-pos -> ngram -> archaic ->
    difficulty -> quizdef -> quizzable -> book-similarity ->
    author-similarity -> wordnik-pron -> ipa -> embed. This is the whole
    documented sequence from the README's
    "Backfilling definitions" / "Enrichment & scoring" / "Definition-quality
    cleanup" / "Pronunciation audio" / "Semantic distance" sections, chained
    into one command instead of twelve to remember and re-order by hand.
    `load-taxonomy` and `train-fasttext` are deliberately excluded — both
    are one-time/occasional holistic setup, not per-batch maintenance (see
    their own docstrings); Commons/Azure audio steps stay separate too,
    since Commons rate-limits hard and is meant to run for hours unattended
    on its own.

    fill-definitions used to be two separate steps here (refill then deepen)
    that each re-entered the definition cascade from scratch on the same
    blank words — now one pass per word at whatever depth --deepen-web
    allows, gated by --recheck-after-days so a permanently-undefined tail
    doesn't get re-ground through Wordnik/web-search on every single run.

    Every step runs incrementally (only-missing / blank-only / not-refetch),
    including forcing classify's only_missing (its own default is False) — so
    a re-run after everything's caught up is fast. The FIRST run against a
    corpus with a real backlog is not: classify and quizdef load a local LLM
    and call it per word, so catching up ~19k words there is the dominant
    cost, likely hours. That cost is paid once; every later run only touches
    the new batch's words. Use the --skip-* flags to defer the slow steps to
    run separately/overnight instead."""
    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    cfg = Config()
    if model:
        cfg.model_path = str(model)

    if not (skip_fill_definitions or skip_refill or skip_deepen):
        with console.status("[bold]Backfilling definitions…"):
            stats = db.fill_definitions(conn, schema, limit=limit, use_web=deepen_web,
                                        model_path=str(model) if model else None,
                                        recheck_after_days=recheck_after_days)
        console.print(f"[green]✓[/green] fill-definitions: [bold]{stats['defined']}[/bold]/{stats['attempted']} "
                      f"defined ({stats['still_undefined']} scored for validity, still undefined, "
                      f"{stats.get('cast_out', 0)} cast out as symbol/proper-noun-only)")
    else:
        console.print("[dim]fill-definitions skipped.[/dim]")

    if not skip_classify:
        from .classify import classify_and_store
        with console.status("[bold]Classifying USAS domains…"):
            stats = classify_and_store(conn, schema, cfg, limit, only_missing=True, batch=None)
        console.print(f"[green]✓[/green] classify: [bold]{stats['classified']}[/bold]/{stats['words']} words "
                      f"-> {stats['assignments']} category assignments")
    else:
        console.print("[dim]classify skipped.[/dim]")

    if not skip_normalize_pos:
        stats = db.normalize_word_pos(conn, schema, limit=limit)
        console.print(f"[green]✓[/green] normalize-pos: [bold]{stats['changed']}[/bold]/{stats['words']} words updated")
    else:
        console.print("[dim]normalize-pos skipped.[/dim]")

    if not skip_ngram:
        stats = db.fetch_ngrams(conn, schema, only_missing=True, limit=limit)
        console.print(f"[green]✓[/green] ngram: fetched [bold]{stats['fetched']}[/bold]/{stats['words']} "
                      f"({stats['in_corpus']} in corpus, {stats['failed']} failed)")
    else:
        console.print("[dim]ngram skipped.[/dim]")

    if not skip_archaic:
        dist = db.compute_archaic(conn, schema, limit=limit)
        total = sum(dist.values())
        parts = ", ".join(f"{k} {v}" for k, v in sorted(dist.items()))
        console.print(f"[green]✓[/green] archaic flags set on [bold]{total}[/bold] words — {parts}")
    else:
        console.print("[dim]archaic skipped.[/dim]")

    if not skip_difficulty:
        stats = db.compute_difficulty(conn, schema, limit=limit)
        console.print(f"[green]✓[/green] difficulty set on [bold]{stats['words']}[/bold] words "
                      f"(mean {stats['mean']}, median {stats['median']})")
    else:
        console.print("[dim]difficulty skipped.[/dim]")

    if not skip_quizdef:
        with console.status("[bold]Building quiz-safe definitions…"):
            stats = db.compute_quiz_definitions(conn, schema, cfg, only_missing=True, limit=limit)
        console.print(f"[green]✓[/green] quiz defs: [bold]{stats['words']}[/bold] words "
                      f"({stats['clean']} clean, {stats['rewritten']} rewritten, {stats['redacted']} redacted)")
    else:
        console.print("[dim]quizdef skipped.[/dim]")

    if not skip_quizzable:
        dist = db.compute_quizzable(conn, schema, limit=limit)
        console.print(f"[green]✓[/green] quizzable: [bold]{dist.get('quizzable',0)}[/bold] quizzable, "
                      f"{dist.get('excluded',0)} excluded")
    else:
        console.print("[dim]quizzable skipped.[/dim]")

    if not skip_book_similarity:
        with console.status("[bold]Computing book vocabulary overlap…"):
            stats = db.compute_book_similarity(conn, schema, limit=limit)
        console.print(f"[green]✓[/green] book-similarity: [bold]{stats['books']}[/bold] books "
                      f"-> [bold]{stats['pairs_stored']}[/bold] related-book pairs stored")
    else:
        console.print("[dim]book-similarity skipped.[/dim]")

    if not skip_author_similarity:
        with console.status("[bold]Computing author vocabulary overlap…"):
            stats = db.compute_author_similarity(conn, schema, limit=limit)
        console.print(f"[green]✓[/green] author-similarity: [bold]{stats['authors']}[/bold] authors "
                      f"-> [bold]{stats['pairs_stored']}[/bold] related-author pairs stored")
    else:
        console.print("[dim]author-similarity skipped.[/dim]")

    if not skip_wordnik:
        stats = db.fetch_wordnik_pronunciations(conn, schema, only_missing=True, limit=limit)
        if "error" in stats:
            console.print(f"[red]✗[/red] wordnik-pron: {stats['error']}")
        else:
            console.print(f"[green]✓[/green] wordnik-pron: [bold]{stats.get('words',0)}[/bold] checked — " +
                          ", ".join(f"{v} {k}" for k, v in stats.items() if k != "words"))
    else:
        console.print("[dim]wordnik-pron skipped.[/dim]")

    if not skip_ipa:
        try:
            stats = db.compute_ipa(conn, schema, dump_path=dump_path, only_missing=True, limit=limit)
        except FileNotFoundError as exc:
            console.print(f"[red]✗[/red] ipa: {exc}")
        else:
            backfilled = (stats.get("backfilled_kaikki", 0) + stats.get("backfilled_wordnik", 0)
                          + stats.get("backfilled_local_wiktionary", 0))
            corrected = (stats.get("corrected_kaikki", 0) + stats.get("corrected_wordnik", 0)
                         + stats.get("corrected_local_wiktionary", 0))
            console.print(f"[green]✓[/green] ipa: {stats.get('total',0)} words — "
                          f"[bold]{stats.get('already_valid',0)}[/bold] already valid, "
                          f"{backfilled} backfilled, {corrected} corrected, "
                          f"{stats.get('cleared_no_replacement',0)} cleared (no valid source found), "
                          f"{stats.get('unresolved',0)} still unresolved")
    else:
        console.print("[dim]ipa skipped.[/dim]")

    if not skip_embed:
        with console.status("[bold]Embedding definitions…"):
            stats = db.compute_definition_embeddings(conn, schema, only_missing=True, limit=limit)
        console.print(f"[green]✓[/green] definition embed: [bold]{stats['embedded']}[/bold]/{stats['words']} "
                      f"embedded ({stats['skipped_no_text']} skipped, no text)")
        if fasttext_model.exists():
            with console.status("[bold]Embedding word forms (FastText)…"):
                stats = db.compute_fasttext_embeddings(conn, schema, model_path=str(fasttext_model),
                                                       only_missing=True, limit=limit)
            console.print(f"[green]✓[/green] fasttext embed: [bold]{stats['embedded']}[/bold]/{stats['words']} embedded")
        else:
            console.print(f"[dim]fasttext embed skipped — {fasttext_model} not found "
                          "(run `concordance train-fasttext` first).[/dim]")
    else:
        console.print("[dim]embed skipped.[/dim]")

    conn.close()


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


@app.command("create-admin")
def create_admin(
    username: str = typer.Argument(..., help="Login username for the new admin account."),
    schema: str = typer.Option(db.DEFAULT_SCHEMA, "--schema", help="Postgres schema."),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="Overrides DATABASE_URL / .env."),
) -> None:
    """One-off: seed an admin-flagged users row for the webapp's own login (separate
    from Cloudflare Access). Prompts for a password via getpass, never as an argument
    — passing it on the command line would land it in shell history."""
    import getpass
    import sys
    from pathlib import Path

    # `webapp` isn't an installed package (unlike `concordance`) -- it's only
    # importable when something puts the repo root on sys.path for you
    # (pytest's rootdir insertion, uvicorn's --app-dir .). The installed
    # `concordance` console-script doesn't, so this command has to do it itself.
    _repo_root = str(Path(__file__).resolve().parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from webapp.backend import auth

    password = getpass.getpass("Password: ")
    if len(password) < 8:
        console.print("[red]✗[/red] password must be at least 8 characters"); raise typer.Exit(code=1)
    if password != getpass.getpass("Confirm password: "):
        console.print("[red]✗[/red] passwords didn't match"); raise typer.Exit(code=1)

    try:
        conn = db.connect(database_url)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗[/red] cannot connect: {exc}"); raise typer.Exit(code=1)
    db.apply_schema(conn, schema)
    s = db._safe_schema(schema)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {s}.users (username, password_hash, is_admin) VALUES (%s,%s,true)",
                (username, auth.hash_password(password)),
            )
        conn.commit()
    except psycopg.errors.UniqueViolation:
        conn.rollback()
        console.print(f"[red]✗[/red] username {username!r} already exists"); raise typer.Exit(code=1)
    finally:
        conn.close()
    console.print(f"[green]✓[/green] admin account [bold]{username}[/bold] created")


if __name__ == "__main__":
    app()
