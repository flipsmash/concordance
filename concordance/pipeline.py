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


# A cached verdict (db.fetch_known_verdicts) -> the (verdict, reject_reason) it
# resolves to. 'keep' becomes a survivor that skips the judge but still gets
# enriched + linked to the new book; the two drop kinds keep their distinct
# reasons so the rejected_word row records *why* (human prune vs judge reject).
_VERDICT_MAP = {
    "keep":   (Verdict.KEEP, None),
    "pruned": (Verdict.DROP, RejectReason.ALREADY_KNOWN),
    "reject": (Verdict.DROP, RejectReason.NOT_INTERESTING),
}


def apply_known_verdicts(candidates: dict[str, Candidate], known: dict[str, str]) -> dict[str, int]:
    """Pre-mark every candidate whose verdict is already known from earlier
    books (see db.fetch_known_verdicts) so the LLM judge is skipped for it.
    Returns per-kind counts. Pure — no DB access — so it's unit-testable."""
    counts = {"keep": 0, "pruned": 0, "reject": 0}
    for c in candidates.values():
        if c.verdict is not None:
            continue
        kind = known.get(c.lemma)
        if kind is None:
            continue
        c.verdict, c.reject_reason = _VERDICT_MAP[kind]
        counts[kind] += 1
    return counts


def process(book: str | Path, cfg: Config, console: Console | None = None,
            schema: str = db.DEFAULT_SCHEMA, *, nlp=None, gate=None, judge_obj=None,
            ) -> tuple[list[Candidate], list[Candidate]]:
    """Extract -> filter -> judge -> enrich a book, returning (kept, rejected).
    Shared by `run` (writes CSVs for hand-editing) and `ingest` (writes straight
    to Postgres) — everything through enrichment is identical; only what
    happens with the result differs.

    Requires a live DATABASE_URL for the local Wiktionary dump (vocab.wiktionary,
    ~500k terms) — checked first in the validity gate and tried first during
    enrichment, both because it's free (no network) and because unlike every
    other authority here it carries no "Proper noun" POS at all, so it doesn't
    get to vouch for real names the way the frequency-based checks do.

    `nlp`, `gate`, `judge_obj` may be pre-built and passed in (a batch run builds
    each once and reuses it across every book, instead of reloading the ~9GB
    judge model + spaCy + the validity corpora per book); each is lazily built
    here when omitted, so a one-off single-book call needs nothing extra."""
    console = console or Console()
    book = Path(book)
    conn = db.connect()

    with console.status("[bold]Extracting text…"):
        chapters = extract.extract(book)
        for ch in chapters:
            ch.text = clean.clean(ch.text)
    console.print(f"Extracted [bold]{len(chapters)}[/bold] section(s) from {book.name}.")

    with console.status("[bold]Tokenizing & lemmatizing…"):
        candidates = tokenize.tokenize(chapters, nlp=nlp)
    console.print(f"Found [bold]{len(candidates)}[/bold] distinct lemmas.")

    lexicon = localdict.build_lexicon(conn, set(candidates.keys()))

    # --- deterministic filters -------------------------------------------
    floor.apply_floor(candidates, cfg)

    # Cross-book verdict cache: a word already kept/pruned/judge-rejected in an
    # earlier book has a known verdict (the judge input is purely lemma-derived),
    # so pre-mark it here and never spend the LLM on it again. Applied AFTER the
    # floor so cached rows still carry a real zipf (every cached lemma is
    # zipf<floor already — it reached the judge before — so the floor never
    # touches them; it just fills in zipf). Cached-keeps become survivors that
    # skip the judge but still get enriched + linked to this book below.
    known = db.fetch_known_verdicts(conn, schema)
    vc = apply_known_verdicts(candidates, known)
    if any(vc.values()):
        console.print(f"[dim]judge skipped for {sum(vc.values())} already-decided word(s) "
                      f"({vc['keep']} kept, {vc['reject']} rejected, {vc['pruned']} pruned) "
                      "from earlier books.[/dim]")

    propernouns.strip_proper_nouns(candidates, cfg)
    with console.status("[bold]Checking validity…"):
        validity.apply_validity(candidates, cfg, local_dict=lexicon, gate=gate)
    survivors = [c for c in candidates.values() if c.verdict in (Verdict.KEEP, Verdict.UNSURE)]
    console.print(f"[bold]{len(survivors)}[/bold] candidates survived the floor + validity gate.")

    # --- LLM interestingness judge ---------------------------------------
    # Only genuinely-new lemmas (no cached verdict) reach the model; cached-keeps
    # keep their KEEP verdict and flow to enrichment/shortlist without a call.
    newly = [c for c in candidates.values() if c.lemma not in known]
    with console.status("[bold]Judging interestingness…"):
        (judge_obj or judge.get_judge(cfg)).judge(newly)
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
