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


def retire(conn, schema: str, keeper: Book, dup: Book, *, by: str, reason: str,
           spread: dict[int, list[int]] | None = None) -> int:
    """Snapshot and remove book `dup`, keeping `keeper`. Returns the
    book_retirement id. Doesn't commit. `spread` ({book id: [word ids]})
    sends dup's word links to several kept books instead of all to keeper
    (a compilation retired in favour of the books it contains); they are
    recorded in the snapshot so restore takes them off again."""
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
        cur.execute(f"""SELECT coalesce(jsonb_agg(to_jsonb(p)), '[]') FROM {s}.force_map_point p
                        WHERE entity = 'book' AND key = %s""", (str(dup.id),))
        snap["rows"]["force_map_point"] = cur.fetchone()[0]
        cur.execute(f"""SELECT coalesce(jsonb_agg(to_jsonb(e)), '[]') FROM {s}.force_map_edge e
                        WHERE entity = 'book' AND (key_a = %(k)s OR key_b = %(k)s)""", {"k": str(dup.id)})
        snap["rows"]["force_map_edge"] = cur.fetchone()[0]

        # its words stay attached to a book: copy the links the keeper lacks
        added: list[int] = []
        if spread is None:
            cur.execute(f"""INSERT INTO {s}.word_book (word_id, book_id)
                            SELECT word_id, %s FROM {s}.word_book WHERE book_id = %s
                            ON CONFLICT DO NOTHING RETURNING word_id""", (keeper.id, dup.id))
            added = [r[0] for r in cur.fetchall()]
        else:
            snap["added_elsewhere"] = {}
            for bid, words in spread.items():
                cur.execute(f"""INSERT INTO {s}.word_book (word_id, book_id)
                                SELECT unnest(%s::int[]), %s ON CONFLICT DO NOTHING RETURNING word_id""",
                            (words, bid))
                got = [r[0] for r in cur.fetchall()]
                if got:
                    snap["added_elsewhere"][str(bid)] = got
        cur.execute(f"DELETE FROM {s}.corpus_map_point WHERE entity = 'book' AND key = %s", (str(dup.id),))
        cur.execute(f"DELETE FROM {s}.force_map_point WHERE entity = 'book' AND key = %s", (str(dup.id),))
        cur.execute(f"DELETE FROM {s}.force_map_edge WHERE entity = 'book' AND (key_a = %(k)s OR key_b = %(k)s)",
                    {"k": str(dup.id)})
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
        for bid, words in (snap.get("added_elsewhere") or {}).items():
            cur.execute(f"DELETE FROM {s}.word_book WHERE book_id = %s AND word_id = ANY(%s)", (int(bid), words))
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


# --- third pass: the same text under another title, and compilations ----------
#
# Titles can't see "A Christmas Carol" and "A Christmas Carol in Prose; Being
# a Ghost Story of Christmas" as one book, or a short-story collection as its
# stories. The texts can. Each archived text is cut into overlapping 8-word
# runs; a fixed 1-in-64 sample of their hashes (the same runs are sampled in
# every book) is the book's sketch, and the share of A's sketch found in B's
# estimates how much of A's text B holds. Runs found in more than MAX_DF books
# are quotations and formulae, not shared text, and are left out. Two
# different works by one author share almost no 8-word runs; a book and its
# reprint share nearly all.

SHINGLE = 8
SAMPLE = 64
MAX_DF = 40
SAME_TEXT = 0.8          # both ways: the same text
INSIDE = 0.8             # one way: A is inside B
SPLIT_PART = 0.9         # a volume of a work its whole edition holds
COVERED = 0.95           # a compilation its contained books cover this much of
SKETCH_CACHE = "data/book_sketches.pkl"
_P = 1099511628211
_WORD = re.compile(r"[a-z]+")
_ORD = r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)"
_NUM = r"(?:\d+|[ivxlc]+|one|two|three|four|five|six|seven|eight|nine|ten)"
_VOLUME_MARK = re.compile(
    rf"\b(?:volume|vol\.?|v\.?|part|book|series)\s*{_NUM}\b|\b{_ORD}\s+(?:part|volume|series|book)\b"
    rf"|\bvolumes?\s+\d+\s*-\s*\d+|\b(?:\d+|[ivx]+)\.?\s*$", re.I)
_TITLE_FILLER = {"the", "a", "an", "complete", "of", "and"}


def sketch_text(path: str):
    """The sampled 8-word-run hashes of one archived text (numpy uint64,
    sorted, unique); None if the file can't be read."""
    import zlib

    import numpy as np
    try:
        text = open(path, errors="replace").read().lower()
    except OSError:
        return None
    toks = _WORD.findall(text)
    n = len(toks) - SHINGLE + 1
    if n <= 0:
        return np.zeros(0, np.uint64)
    h = np.fromiter((zlib.crc32(t.encode()) for t in toks), dtype=np.uint64, count=len(toks))
    acc = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(SHINGLE):
            acc = acc * np.uint64(_P) + h[j: n + j]
        acc ^= acc >> np.uint64(29)                     # mix, so `% SAMPLE` samples evenly
        acc *= np.uint64(0xBF58476D1CE4E5B9)
        acc ^= acc >> np.uint64(32)
    return np.unique(acc[acc % np.uint64(SAMPLE) == 0])


def _sketch_job(args):
    bid, path = args
    return bid, sketch_text(path)


def sketches(conn, schema: str, *, cache_path: str = SKETCH_CACHE, workers: int = 8, log=print) -> dict:
    """{book id: sketch} for every book with an archived .txt, reusing the
    cache for files unchanged since (path, size, mtime)."""
    import os
    import pickle
    from multiprocessing import Pool
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, archive_path FROM {s}.book WHERE archive_path LIKE '%%.txt'")
        rows = cur.fetchall()
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        cache = {}
    out, todo, stamps = {}, [], {}
    for bid, path in rows:
        try:
            st = os.stat(path)
        except OSError:
            continue
        stamps[bid] = (path, st.st_size, int(st.st_mtime))
        hit = cache.get(bid)
        if hit and hit[0] == stamps[bid]:
            out[bid] = hit[1]
        else:
            todo.append((bid, path))
    if todo:
        log(f"  sketching {len(todo):,} text(s) ({len(out):,} cached)")
        with Pool(workers) as pool:
            for bid, sk in pool.imap_unordered(_sketch_job, todo, chunksize=20):
                if sk is not None:
                    out[bid] = sk
    new_cache = {bid: (stamps[bid], sk) for bid, sk in out.items()}
    if todo:
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(new_cache, f)
        os.replace(tmp, cache_path)
    return out


def text_overlaps(sk: dict, min_share: float = 0.3, min_shared: int = 5) -> list[tuple[int, int, float, float]]:
    """[(a, b, share of a's text in b, share of b's text in a)] for every
    pair sharing at least `min_share` of one of them."""
    from collections import Counter

    import numpy as np
    ids = np.array(sorted(sk), dtype=np.int64)
    if not len(ids):
        return []
    sizes = np.array([len(sk[b]) for b in ids], dtype=np.int64)
    hashes = np.concatenate([sk[b] for b in ids])
    owner = np.concatenate([np.full(len(sk[b]), i, np.int32) for i, b in enumerate(ids)])
    order = np.argsort(hashes, kind="stable")
    hashes, owner = hashes[order], owner[order]
    starts = np.flatnonzero(np.r_[True, hashes[1:] != hashes[:-1]])
    df = np.diff(np.r_[starts, len(hashes)])
    np.subtract.at(sizes, owner[np.repeat(df > MAX_DF, df)], 1)     # formulae don't count anywhere
    shared: Counter = Counter()
    keep = (df >= 2) & (df <= MAX_DF)
    for st, d in zip(starts[keep], df[keep]):
        members = sorted(set(owner[st:st + d].tolist()))
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                shared[(a, b)] += 1
    out = []
    for (a, b), n in shared.items():
        if n < min_shared:
            continue
        ab, ba = n / max(sizes[a], 1), n / max(sizes[b], 1)
        if max(ab, ba) >= min_share:
            out.append((int(ids[a]), int(ids[b]), round(ab, 3), round(ba, 3)))
    return out


def _books(conn, schema: str) -> dict[int, Book]:
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT b.id, b.title, b.author, b.word_count,
                               (SELECT count(*) FROM {s}.word_book wb WHERE wb.book_id = b.id)
                        FROM {s}.book b""")
        return {r[0]: Book(*r) for r in cur.fetchall()}


def find_same_text(books: dict[int, Book], overlaps) -> list[tuple[Book, list[Book]]]:
    """[(keeper, [copies to remove])]: groups of books each holding at least
    SAME_TEXT of the other's text, whatever their titles; the copy with more
    linked words stays."""
    parent: dict[int, int] = {}

    def root(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for a, b, ab, ba in overlaps:
        if min(ab, ba) >= SAME_TEXT and a in books and b in books:
            parent[root(a)] = root(b)
    groups: dict[int, list[Book]] = defaultdict(list)
    for x in list(parent):
        groups[root(x)].append(books[x])
    out = []
    for members in groups.values():
        members = sorted(members, key=lambda m: (-m.linked_words, m.id))
        if len(members) > 1:
            out.append((members[0], members[1:]))
    return sorted(out, key=lambda g: g[0].title.lower())


def find_containers(books: dict[int, Book], overlaps, sk: dict, gone: dict[int, int] | None = None):
    """After same-text copies are gone (`gone`: removed id -> keeper id):

      ("split", whole, [volumes]) -- every book inside `whole` is a volume
          or partial edition of it ("Volume 07", "V2", "(Complete)"): the
          whole stays and its volumes go (like the --contained pass);
      ("compilation", container, [books], coverage) -- `container` holds
          distinct works that, together, cover COVERED of its text: the
          compilation goes and they stay.

    A container whose parts are distinct works but don't cover it (an
    "Entire Works" holding things no other book has) is left alone."""
    import numpy as np
    gone = gone or {}
    inside: dict[int, dict[int, float]] = defaultdict(dict)
    for a, b, ab, ba in overlaps:
        a, b = gone.get(a, a), gone.get(b, b)
        if a == b or a not in books or b not in books:
            continue
        if ab >= INSIDE and ba < SAME_TEXT:
            inside[b][a] = max(ab, inside[b].get(a, 0))
        elif ba >= INSIDE and ab < SAME_TEXT:
            inside[a][b] = max(ba, inside[a].get(b, 0))
    def coverage(big, ids):
        mine = sk.get(big)
        if mine is None or not len(mine):
            return 0.0
        covered = np.zeros(len(mine), bool)
        for p in ids:
            if p in sk:
                covered |= np.isin(mine, sk[p])
        return float(covered.mean())

    splits, comps = {}, []
    for big, parts in inside.items():
        whole = books[big]
        pieces = {p: share for p, share in parts.items() if _piece_of(books[p].title, whole.title)}
        if pieces and (len(pieces) == len(parts) or coverage(big, pieces) >= COVERED):
            vols = [p for p, share in pieces.items() if share >= SPLIT_PART]
            if vols:
                splits[big] = vols
            continue
        cov = coverage(big, parts)
        if cov >= COVERED:
            comps.append(("compilation", whole, sorted((books[p] for p in parts), key=lambda b: -(b.word_count or 0)),
                          round(cov, 3)))
    # a whole that is itself a volume of a bigger whole (Life on the Mississippi,
    # Part 9 holding Parts 10-12) isn't one; each volume goes once, to the
    # biggest whole that holds it
    volume_of: dict[int, int] = {}
    for big in sorted(splits, key=lambda b: -(books[b].word_count or 0)):
        for v in splits[big]:
            volume_of.setdefault(v, big)
    out = []
    for big, vols in splits.items():
        if big in volume_of:
            continue
        mine = [books[v] for v in vols if volume_of.get(v) == big]
        if mine:
            out.append(("split", books[big], sorted(mine, key=lambda b: b.title)))
    return out + comps


def _title_tokens(title: str) -> list[str]:
    return [t for t in _norm(title).split() if t not in _TITLE_FILLER]


def _has_run(tokens: list[str], run: list[str]) -> bool:
    """`run` appears in `tokens` as consecutive whole words."""
    n = len(run)
    return any(tokens[i:i + n] == run for i in range(len(tokens) - n + 1))


def _piece_of(part: str, whole: str) -> bool:
    """Is `part` a volume or section of `whole` by its title: a volume mark
    ("Volume 07", "First Part", "Series Two", a trailing "II" -- not a bare
    "Complete", which titles distinct works too), the whole's
    title inside it ("My Mark Twain (from Literary Friends and Acquaintance)"),
    or the same opening words ("Petty Troubles of Married Life, Second Part")?"""
    if _VOLUME_MARK.search(part):
        return True
    p, w = _title_tokens(part), _title_tokens(whole)
    if not p or not w:
        return False
    main = _title_tokens(re.split(r"[;:(]", whole)[0])     # the title before its subtitle
    if any(len(t) >= 2 and _has_run(p, t) for t in (w, main)):
        return True
    common = 0
    for a, b in zip(p, w):
        if a != b:
            break
        common += 1
    return common >= 3 or (common >= 2 and common >= 0.6 * min(len(p), len(w)))


def spread_links(conn, schema: str, container: Book, parts: list[Book]) -> dict[int, list[int]]:
    """Where each of a compilation's word links should go when it is retired:
    nowhere if a contained book already has the word; else the (largest)
    contained book whose text has the word; else the largest contained book."""
    from pathlib import Path
    s = _safe_schema(schema)
    ids = [p.id for p in parts]
    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, lower(w.lemma) FROM {s}.word_book wb JOIN {s}.word w ON w.id = wb.word_id
                        WHERE wb.book_id = %s AND NOT EXISTS (
                            SELECT 1 FROM {s}.word_book x WHERE x.word_id = w.id AND x.book_id = ANY(%s))""",
                    (container.id, ids))
        orphans = cur.fetchall()
        cur.execute(f"SELECT id, archive_path FROM {s}.book WHERE id = ANY(%s)", (ids,))
        paths = dict(cur.fetchall())
    texts = {}
    for p in parts:
        try:
            texts[p.id] = Path(paths.get(p.id) or "").read_text(errors="replace").lower()
        except OSError:
            texts[p.id] = ""
    out: dict[int, list[int]] = defaultdict(list)
    for wid, lemma in orphans:
        pat = re.compile(r"\b" + re.escape(lemma))
        home = next((p.id for p in parts if pat.search(texts[p.id])), parts[0].id)
        out[home].append(wid)
    return dict(out)


def _exists(conn, schema: str, book_id: int) -> bool:
    with conn.cursor() as cur:
        return _book_exists(cur, _safe_schema(schema), book_id)
