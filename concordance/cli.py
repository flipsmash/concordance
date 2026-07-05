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


if __name__ == "__main__":
    app()
