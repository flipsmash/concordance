"""Duplicate books: find them, take one copy out, and put it back on request.

Gutenberg often carries one work twice under near-identical titles --
"The Divine Comedy by Dante, Illustrated, Paradise (Complete)" and "...,
Paradise, Complete" -- and both were ingested. This first, conservative pass
treats two books as the same only when all of these hold:

  - the same title and the same author once case, punctuation and spacing
    are ignored;
  - a real author (never "Various", "Anonymous"...: two anthologies can
    share a title);
  - word counts within 3% of each other (both known).

The copy with more linked words is kept. Removing the other is reversible:
its row and every row that referenced it (word links, rejections, genres,
fame, stats, similarity, clusters, map points) are snapshotted into
book_retirement before the delete, and its word links are first copied onto
the kept copy, so no word loses its only book. `restore` puts it all back
and takes those copied links off again. book_retirement.status is the
switch: removed | restored.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from .db import _safe_schema
from .db.books import PLACEHOLDER_AUTHORS

WORD_COUNT_TOLERANCE = 0.03

# Every table that points at a book, with the column(s) that do. Snapshotted
# and deleted along with the book (the foreign keys would cascade anyway).
_DEPENDENTS = (
    ("word_book", ("book_id",)),
    ("rejected_word", ("book_id",)),
    ("book_genre", ("book_id",)),
    ("book_fame", ("book_id",)),
    ("book_cluster_fame", ("book_id",)),
    ("book_cluster", ("book_id",)),
    ("book_stats", ("book_id",)),
    ("book_similarity", ("book_a_id", "book_b_id")),
)


def _norm(text: str | None) -> str:
    return re.sub(r"[\W_]+", " ", (text or "").lower()).strip()


@dataclass
class Book:
    id: int
    title: str
    author: str | None
    word_count: int | None
    linked_words: int


def find_duplicates(conn, schema: str) -> list[tuple[Book, list[Book]]]:
    """[(the copy to keep, [the copies to remove])] under the rules above."""
    s = _safe_schema(schema)
    placeholders = {_norm(a) for a in PLACEHOLDER_AUTHORS}
    with conn.cursor() as cur:
        cur.execute(f"""SELECT b.id, b.title, b.author, b.word_count,
                               (SELECT count(*) FROM {s}.word_book wb WHERE wb.book_id = b.id)
                        FROM {s}.book b""")
        books = [Book(*r) for r in cur.fetchall()]
    groups: dict[tuple[str, str], list[Book]] = defaultdict(list)
    for b in books:
        author = _norm(b.author)
        if author and author not in placeholders:
            groups[(_norm(b.title), author)].append(b)
    out = []
    for members in groups.values():
        if len(members) < 2 or any(not m.word_count for m in members):
            continue
        counts = [m.word_count for m in members]
        if (max(counts) - min(counts)) / max(counts) > WORD_COUNT_TOLERANCE:
            continue
        members.sort(key=lambda m: (-m.linked_words, m.id))
        out.append((members[0], members[1:]))
    return sorted(out, key=lambda g: g[0].title.lower())


def retire(conn, schema: str, keeper: Book, dup: Book, *, by: str, reason: str) -> int:
    """Snapshot and remove book `dup`, keeping `keeper`. Returns the
    book_retirement id. Doesn't commit."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT to_jsonb(b) FROM {s}.book b WHERE id = %s", (dup.id,))
        snap = {"book": cur.fetchone()[0], "rows": {}}
        for table, cols in _DEPENDENTS:
            where = " OR ".join(f"{c} = %(id)s" for c in cols)
            cur.execute(f"SELECT coalesce(jsonb_agg(to_jsonb(t)), '[]') FROM {s}.{table} t WHERE {where}", {"id": dup.id})
            snap["rows"][table] = cur.fetchone()[0]
        cur.execute(f"""SELECT coalesce(jsonb_agg(to_jsonb(p)), '[]') FROM {s}.corpus_map_point p
                        WHERE entity = 'book' AND key = %s""", (str(dup.id),))
        snap["rows"]["corpus_map_point"] = cur.fetchone()[0]

        # its words stay attached to a book: copy the links the keeper lacks
        cur.execute(f"""INSERT INTO {s}.word_book (word_id, book_id)
                        SELECT word_id, %s FROM {s}.word_book WHERE book_id = %s
                        ON CONFLICT DO NOTHING RETURNING word_id""", (keeper.id, dup.id))
        added = [r[0] for r in cur.fetchall()]
        cur.execute(f"DELETE FROM {s}.corpus_map_point WHERE entity = 'book' AND key = %s", (str(dup.id),))
        cur.execute(f"UPDATE {s}.book_upload SET book_id = %s WHERE book_id = %s", (keeper.id, dup.id))
        cur.execute(f"DELETE FROM {s}.book WHERE id = %s", (dup.id,))   # dependents cascade
        cur.execute(f"""INSERT INTO {s}.book_retirement
                            (book_id, keeper_id, title, author, reason, snapshot, added_links, retired_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (dup.id, keeper.id, dup.title, dup.author, reason, json.dumps(snap), added, by))
        return cur.fetchone()[0]


def restore(conn, schema: str, retirement_id: int) -> dict:
    """Put a removed book back exactly as it was and take the copied word
    links off its keeper. Doesn't commit. Returns {"book_id", "title"}."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT book_id, keeper_id, snapshot, added_links, status FROM {s}.book_retirement
                        WHERE id = %s FOR UPDATE""", (retirement_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"no retirement {retirement_id}")
        book_id, keeper_id, snap, added, status = row
        if status != "removed":
            raise ValueError(f"retirement {retirement_id} was already restored")
        cols = _insertable(cur, s, "book")
        cur.execute(f"INSERT INTO {s}.book ({cols}) SELECT {cols} FROM jsonb_populate_record(NULL::{s}.book, %s)",
                    (json.dumps(snap["book"]),))
        for table, rows in snap["rows"].items():
            if not rows:
                continue
            if table == "book_similarity":   # the other book of a pair may itself be gone now
                rows = [r for r in rows if _book_exists(cur, s, r["book_a_id"]) and _book_exists(cur, s, r["book_b_id"])]
            cols = _insertable(cur, s, table)
            cur.execute(f"""INSERT INTO {s}.{table} ({cols})
                            SELECT {cols} FROM jsonb_populate_recordset(NULL::{s}.{table}, %s)
                            ON CONFLICT DO NOTHING""", (json.dumps(rows),))
        if added:
            cur.execute(f"DELETE FROM {s}.word_book WHERE book_id = %s AND word_id = ANY(%s)", (keeper_id, added))
        cur.execute(f"UPDATE {s}.book_retirement SET status = 'restored', restored_at = now() WHERE id = %s",
                    (retirement_id,))
        return {"book_id": book_id, "title": snap["book"]["title"]}


def _insertable(cur, s: str, table: str) -> str:
    """The table's columns minus generated ones (rejected_word.lemma_lc),
    which can't be written."""
    cur.execute("""SELECT column_name FROM information_schema.columns
                   WHERE table_schema = %s AND table_name = %s AND is_generated = 'NEVER'
                   ORDER BY ordinal_position""", (s, table))
    return ", ".join(f'"{r[0]}"' for r in cur.fetchall())


def _book_exists(cur, s: str, book_id: int) -> bool:
    cur.execute(f"SELECT 1 FROM {s}.book WHERE id = %s", (book_id,))
    return cur.fetchone() is not None


# --- second pass: books another book of the same work already contains --------

# "Title — Volume 03", "Title, Part 1", "Title (Complete)", "Title — Complete":
# Gutenberg's multi-volume works, our own book-merge compilations, and its
# one-file complete editions, all under one base title.
_PART = re.compile(r"^(.*?)[\s,;:—–-]*\b(?:volume|vol\.?|part|book)\s+[0-9ivxlc]+\.?\s*$", re.I)
_COMPLETE = re.compile(r"^(.*?)[\s,;:—–(\[-]*\bcomplete\b[)\]\.]*\s*$", re.I)
CONTAINED = 0.95


def _work(title: str) -> tuple[str, bool] | None:
    """(base title, is a "Complete" book) for a volume/part/complete title."""
    if m := _COMPLETE.match(title):
        return _norm(m.group(1)), True
    if m := _PART.match(title):
        return _norm(m.group(1)), False
    return None


def _text_vocab(path: str | None) -> set[str] | None:
    from pathlib import Path
    if not path or not path.endswith(".txt") or not Path(path).exists():
        return None
    return set(re.findall(r"[a-z]{4,}", Path(path).read_text(errors="replace").lower()))


def find_contained(conn, schema: str, threshold: float = CONTAINED) -> list[tuple[Book, Book, float]]:
    """[(keeper, book to remove, share of its text's words found in the
    keeper)] within multi-volume works that have a Complete book: a volume
    whose text a Complete edition already holds, or a partial compilation
    inside a fuller one. Compared on the archived texts' vocabularies --
    their linked words differ with when each was ingested. A volume the
    Complete book lacks (our own compilation missed it) scores low and stays."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT b.id, b.title, b.author, b.word_count, b.archive_path,
                               (SELECT count(*) FROM {s}.word_book wb WHERE wb.book_id = b.id)
                        FROM {s}.book b""")
        rows = cur.fetchall()
    works: dict[tuple[str, str], list] = defaultdict(list)
    for bid, title, author, wc, path, linked in rows:
        w = _work(title)
        if w and _norm(author):
            works[(w[0], _norm(author))].append((Book(bid, title, author, wc, linked), w[1], path))
    out = []
    for members in works.values():
        if not any(is_complete for _, is_complete, _ in members) or len(members) < 2:
            continue
        members.sort(key=lambda m: -(m[0].word_count or 0))
        vocab = {m[0].id: _text_vocab(m[2]) for m in members}
        kept: list[Book] = []
        for book, _, _ in members:
            mine = vocab[book.id]
            best = None
            if mine:
                for k in kept:
                    theirs = vocab[k.id]
                    if theirs and (k.word_count or 0) > (book.word_count or 0):
                        share = len(mine & theirs) / len(mine)
                        if share >= threshold and (best is None or share > best[1]):
                            best = (k, share)
            if best:
                out.append((best[0], book, round(best[1], 3)))
            else:
                kept.append(book)
    return out


def book(conn, schema: str, book_id: int) -> Book:
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT b.id, b.title, b.author, b.word_count,
                               (SELECT count(*) FROM {s}.word_book wb WHERE wb.book_id = b.id)
                        FROM {s}.book b WHERE b.id = %s""", (book_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"no book #{book_id}")
    return Book(*row)
