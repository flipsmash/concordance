"""Command-line entry point.

Two verbs:
  concordance run <book>            extract → filter → judge → enrich a book,
                                    writing <book>.vocab.csv (and a pristine copy
                                    to archive/) for you to hand-edit.
  concordance finalize <vocab.csv>  after you delete the rows you know / dislike,
                                    promote the survivors to master_vocab.csv and
                                    archive the book's files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import Config
from .extract import ScannedPDFError, UnsupportedFormatError
from . import db
from .deepen import define as define_cmd
from .finalize import finalize_file
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


if __name__ == "__main__":
    app()
