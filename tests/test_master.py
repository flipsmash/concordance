"""Master-list promotion + archiving, and the standalone review flow."""

from __future__ import annotations

import csv
from pathlib import Path

from concordance import finalize, master
from concordance.output import VOCAB_COLUMNS


def _vocab_row(word, **over):
    r = {c: "" for c in VOCAB_COLUMNS}
    r["word"] = word
    r["definition"] = f"def of {word}"
    r.update(over)
    return r


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- promote_to_master ----------------------------------------------------

def test_promote_adds_new_words(tmp_path):
    m = tmp_path / "master.csv"
    added, merged = master.promote_to_master(
        [_vocab_row("cangue"), _vocab_row("fuligin")], "BookA", m, today="2026-07-04"
    )
    assert (added, merged) == (2, 0)
    rows = _read(m)
    assert {r["word"] for r in rows} == {"cangue", "fuligin"}
    assert all(r["date_added"] == "2026-07-04" and r["source_book"] == "BookA" for r in rows)


def test_repeat_word_from_new_book_appends_source_not_row(tmp_path):
    m = tmp_path / "master.csv"
    master.promote_to_master([_vocab_row("cangue")], "BookA", m, today="2026-07-01")
    added, merged = master.promote_to_master([_vocab_row("cangue")], "BookB", m, today="2026-07-09")
    assert (added, merged) == (0, 1)
    rows = _read(m)
    assert len(rows) == 1                       # one row per word
    assert rows[0]["source_book"] == "BookA; BookB"
    assert rows[0]["date_added"] == "2026-07-01"  # first date preserved


def test_repeat_word_from_same_book_is_noop(tmp_path):
    m = tmp_path / "master.csv"
    master.promote_to_master([_vocab_row("cangue")], "BookA", m)
    added, merged = master.promote_to_master([_vocab_row("cangue")], "BookA", m)
    assert (added, merged) == (0, 0)
    assert _read(m)[0]["source_book"] == "BookA"


def test_promote_case_insensitive_dedup(tmp_path):
    m = tmp_path / "master.csv"
    master.promote_to_master([_vocab_row("Cangue")], "BookA", m)
    added, merged = master.promote_to_master([_vocab_row("cangue")], "BookB", m)
    assert (added, merged) == (0, 1)
    assert len(_read(m)) == 1


# --- archive_book ---------------------------------------------------------

def test_archive_moves_artifacts_and_book(tmp_path):
    stem = "MyBook"
    (tmp_path / f"{stem}.vocab.csv").write_text("x")
    (tmp_path / f"{stem}.rejected.csv").write_text("x")
    (tmp_path / f"{stem}.epub").write_text("x")
    (tmp_path / "unrelated.txt").write_text("x")
    arch = tmp_path / "archive"
    moved, failed = master.archive_book(tmp_path / f"{stem}.vocab.csv", arch)
    assert failed == []
    assert {p.name for p in moved} == {f"{stem}.vocab.csv", f"{stem}.rejected.csv", f"{stem}.epub"}
    assert (arch / f"{stem}.epub").exists()
    assert not (tmp_path / f"{stem}.vocab.csv").exists()
    assert (tmp_path / "unrelated.txt").exists()   # untouched


def test_book_stem():
    assert master.book_stem(Path("a/b/Shadow.vocab.csv")) == "Shadow"


def test_snapshot_original_copies_before_edit(tmp_path):
    vocab = tmp_path / "Book.vocab.csv"
    vocab.write_text("word\ncangue\n")
    arch = tmp_path / "archive"
    dest = master.snapshot_original(vocab, arch)
    assert dest == arch / "Book.vocab.original.csv"
    assert dest.read_text() == "word\ncangue\n"
    assert vocab.exists()                            # original left in place to edit


# --- finalize (hand-edited CSV: surviving rows = approved) -----------------

def _write_vocab(path: Path, words):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VOCAB_COLUMNS)
        w.writeheader()
        for word in words:
            w.writerow(_vocab_row(word))


def test_finalize_promotes_survivors_and_archives(tmp_path):
    vocab = tmp_path / "Book.vocab.csv"
    (tmp_path / "Book.epub").write_text("x")
    (tmp_path / "Book.rejected.csv").write_text("x")
    # user kept only cangue + fuligin (deleted whisper by hand)
    _write_vocab(vocab, ["cangue", "fuligin"])
    m = tmp_path / "master_vocab.csv"
    arch = tmp_path / "archive"
    finalize.finalize_file(vocab, master_path=m, archive_dir=arch, assume_yes=True)
    assert {r["word"] for r in _read(m)} == {"cangue", "fuligin"}
    assert (arch / "Book.epub").exists()
    assert (arch / "Book.vocab.csv").exists()
    assert (arch / "Book.rejected.csv").exists()
    assert not vocab.exists()                         # moved out of working dir


def test_finalize_reads_excel_mangled_header(tmp_path):
    """Excel renamed the header to Column1..N and appended a stray column — still
    reads every data row positionally and ignores the extra column."""
    vocab = tmp_path / "Book.vocab.csv"
    with vocab.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"Column{i+1}" for i in range(len(VOCAB_COLUMNS) + 1)])  # generic + 1 extra
        # canonical order: word, as_seen, definition, ... , source, then a stray status cell
        w.writerow(["cangue", "cangue", "a collar", "noun", "", "The cangue.", "", "", "", "Wiktionary", "keep"])
        w.writerow(["fuligin", "fuligin", "soot", "noun", "", "The fuligin.", "", "", "", "Wiktionary", "keep"])
    from concordance.finalize import _read_rows
    rows = _read_rows(vocab)
    assert [r["word"] for r in rows] == ["cangue", "fuligin"]
    assert rows[0]["definition"] == "a collar"        # positional mapping intact
    assert "status" not in rows[0]                     # stray column dropped


def test_finalize_cancelled_leaves_everything(tmp_path):
    vocab = tmp_path / "Book.vocab.csv"
    _write_vocab(vocab, ["cangue"])
    m = tmp_path / "master_vocab.csv"
    finalize.finalize_file(vocab, input_fn=lambda *a, **k: "n",
                           master_path=m, archive_dir=tmp_path / "archive")
    assert not m.exists()
    assert vocab.exists()                             # nothing moved


def test_finalize_second_book_merges_source(tmp_path):
    m = tmp_path / "master_vocab.csv"
    a = tmp_path / "BookA.vocab.csv"; _write_vocab(a, ["cangue"])
    finalize.finalize_file(a, master_path=m, archive_dir=tmp_path / "archive", assume_yes=True)
    b = tmp_path / "BookB.vocab.csv"; _write_vocab(b, ["cangue", "fuligin"])
    finalize.finalize_file(b, master_path=m, archive_dir=tmp_path / "archive", assume_yes=True)
    rows = {r["word"]: r["source_book"] for r in _read(m)}
    assert rows["cangue"] == "BookA; BookB"
    assert rows["fuligin"] == "BookB"
