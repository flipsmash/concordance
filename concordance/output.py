"""Stage 10b — write results (§03.10 / §07).

Two CSVs alongside the book:
  <book>.vocab.csv     the words you kept (export-friendly for another app later)
  <book>.rejected.csv  everything cut, with the reason — nothing vanishes silently
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from .model import Candidate, Verdict

VOCAB_COLUMNS = [
    "word", "as_seen", "definition", "part_of_speech", "ipa",
    "sentence", "chapter", "synonyms", "etymology", "source",
]
REJECTED_COLUMNS = ["word", "reason", "detail", "count", "zipf"]


def _atomic_write(path: Path, rows: list[list]) -> None:
    """Write a CSV via a sibling temp file + os.replace. On /mnt/c (drvfs) a file
    open in Excel blocks ``open(path, "w")`` with PermissionError, but an atomic
    rename onto it succeeds — so never open the destination directly.

    utf-8-sig (a BOM) rather than plain utf-8: without it Excel opens an
    accented CSV (maté, gâteaux) using the system codepage instead of UTF-8,
    rendering correctly-encoded characters as mojibake even though the file
    itself is fine."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)
    os.replace(tmp, path)


def write_vocab(path: Path, kept: list[Candidate]) -> None:
    rows = [VOCAB_COLUMNS]
    for c in kept:
        rep = c.representative
        rows.append([
            c.lemma,
            rep.surface if rep else "",
            c.definition,
            (c.part_of_speech or c.pos).lower(),
            c.ipa,
            rep.sentence if rep else "",
            rep.chapter if rep else "",
            "; ".join(c.synonyms),
            c.etymology,
            c.definition_source or ", ".join(c.validity_sources),
        ])
    _atomic_write(path, rows)


def write_rejected(path: Path, rejected: list[Candidate]) -> None:
    rows = [REJECTED_COLUMNS]
    for c in rejected:
        rows.append([
            c.lemma,
            c.reject_reason.value if c.reject_reason else "",
            c.interesting_reason,
            c.count,
            f"{c.zipf:.2f}",
        ])
    _atomic_write(path, rows)


def partition(candidates: dict[str, Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    """Split into (kept, rejected). UNSURE that survived review counts as kept."""
    kept, rejected = [], []
    for c in candidates.values():
        if c.verdict in (Verdict.KEEP, Verdict.UNSURE):
            kept.append(c)
        else:
            rejected.append(c)
    kept.sort(key=lambda c: (c.zipf, c.lemma))       # rarest first
    rejected.sort(key=lambda c: (c.reject_reason.value if c.reject_reason else "", c.lemma))
    return kept, rejected
