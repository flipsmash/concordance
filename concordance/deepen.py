"""`concordance define` — resolve the undefined tail (§03.9 / §05 follow-on).

Runs on an existing <book>.vocab.csv, only touching rows still missing a
definition. For each: try the deep sources (Wordnik, yourdictionary; with --web,
also web-search + grounded LLM extraction). Whatever gets defined is written back
into the vocab CSV. Whatever stays undefined gets a validity estimate (real word
vs OCR/variant/Latin/nonsense), collected into a sibling <book>.undefined.csv you
can triage.

    concordance define "some book.vocab.csv"          # dictionaries only
    concordance define "some book.vocab.csv" --web     # + web-search/LLM last resort
"""

from __future__ import annotations

import csv
import os
import time
from collections import Counter
from pathlib import Path

from rich.console import Console

from . import db, deepdef, dictionary, localdict, validity_score, websearch
from .config import Config
from .finalize import _read_rows
from .model import Candidate, Occurrence, normalize_pos
from .output import VOCAB_COLUMNS

_POLITE = 0.5   # seconds between words — Wordnik's free tier rate-limits hard

UNDEFINED_COLUMNS = [
    "word", "as_seen", "sentence", "chapter",
    "validity_label", "validity_score", "validity_notes", "suggested_correction",
]


def _row_to_candidate(row: dict) -> Candidate:
    c = Candidate(lemma=row["word"], pos=(row.get("part_of_speech") or "NOUN").upper())
    if row.get("sentence"):
        c.occurrences.append(Occurrence(sentence=row["sentence"],
                                        chapter=row.get("chapter", ""),
                                        surface=row.get("as_seen") or row["word"]))
    return c


def _fill(row: dict, cand: Candidate) -> None:
    row["definition"] = cand.definition
    if cand.part_of_speech:
        row["part_of_speech"] = normalize_pos(cand.part_of_speech)
    row["source"] = cand.definition_source


def _load_llm(model_path: str | None, console: Console):
    cfg = Config()
    mp = model_path or cfg.model_path
    if not mp or not Path(mp).exists():
        console.print("[yellow]--web:[/yellow] model not found — skipping web extraction.")
        return None
    try:
        from llama_cpp import Llama
        console.print("[dim]loading model for web extraction…[/dim]")
        return Llama(model_path=mp, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)
    except Exception as exc:                     # noqa: BLE001
        console.print(f"[yellow]--web:[/yellow] could not load model ({exc}) — skipping.")
        return None


def define(vocab_path: Path, console: Console | None = None,
           use_web: bool = False, model_path: str | None = None) -> tuple[int, int]:
    """Returns (defined, still_undefined)."""
    console = console or Console()
    vocab_path = Path(vocab_path)
    rows = _read_rows(vocab_path)
    undefined = [r for r in rows if not (r.get("definition") or "").strip()]
    console.print(f"{len(rows)} rows · [bold]{len(undefined)}[/bold] still undefined.")
    if not undefined:
        return 0, 0

    conn = db.connect()
    lexicon = localdict.build_lexicon(conn, {(r.get("word") or "").strip().lower() for r in undefined})
    conn.close()

    session = dictionary.make_session()
    key = deepdef.wordnik_key()
    if not key:
        console.print("[yellow]note:[/yellow] no WORDNIK_API_KEY — skipping Wordnik (yourdictionary only).")
    llm = _load_llm(model_path, console) if use_web else None
    use_web = use_web and llm is not None

    dict_hits = web_hits = 0
    report: list[dict] = []
    with console.status("[bold]Resolving…") as status:
        for i, row in enumerate(undefined, 1):
            cand = _row_to_candidate(row)
            if localdict.enrich(cand, lexicon) or deepdef.deep_enrich(cand, session, key):
                _fill(row, cand)
                dict_hits += 1
            else:
                est = validity_score.estimate(row["word"], session=session, sentence=row.get("sentence", ""))
                # Only spend a web search on words that aren't already obvious junk.
                if use_web and est.label != "likely-artifact" and websearch.define_via_web(cand, llm):
                    _fill(row, cand)
                    web_hits += 1
                else:
                    report.append({
                        "word": row["word"], "as_seen": row.get("as_seen", ""),
                        "sentence": row.get("sentence", ""), "chapter": row.get("chapter", ""),
                        "validity_label": est.label, "validity_score": est.score,
                        "validity_notes": est.notes, "suggested_correction": est.suggestion,
                    })
            status.update(f"[bold]Resolving… {i}/{len(undefined)} · defined {dict_hits + web_hits}")
            time.sleep(_POLITE)

    _write_vocab(vocab_path, rows)
    report_path = None
    if report:
        report.sort(key=lambda r: r["validity_score"])
        report_path = vocab_path.with_name(deepdef_stem(vocab_path) + ".undefined.csv")
        _write_report(report_path, report)

    web_note = f" (+{web_hits} via web)" if use_web else ""
    console.print(f"[green]✓[/green] defined [bold]{dict_hits + web_hits}[/bold]/{len(undefined)}"
                  f"{web_note} · updated {vocab_path.name}")
    if report:
        c = Counter(r["validity_label"] for r in report)
        console.print(
            f"[green]✓[/green] {len(report)} still undefined → {report_path.name}  "
            f"[dim](likely-valid {c['likely-valid']}, uncertain {c['uncertain']}, "
            f"likely-artifact {c['likely-artifact']})[/dim]"
        )
    return dict_hits + web_hits, len(report)


def deepdef_stem(vocab_path: Path) -> str:
    name = vocab_path.name
    return name[: -len(".vocab.csv")] if name.endswith(".vocab.csv") else vocab_path.with_suffix("").stem


def _write_vocab(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=VOCAB_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _write_report(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=UNDEFINED_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
