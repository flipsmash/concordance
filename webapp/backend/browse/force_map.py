"""/api/browse/force-map: the force-directed map of every book, author or
word, as laid out by `concordance force-map` (concordance/force_map.py).

Columnar like /api/browse/corpus-map (parallel arrays; links as index pairs
into them). The word map is ~70k points, so the body is built once per run
and kept in memory, keyed by the run's computed_at, and revalidated by ETag."""

from __future__ import annotations

import json
from typing import Literal

from fastapi import Depends, HTTPException, Request
from fastapi.responses import Response
from webapp.backend import main as _main

from .common import router

_cache: dict[tuple[str, str], tuple[str, bytes]] = {}


def _build(cur, s: str, entity: str, computed_at) -> bytes:
    cur.execute(f"""SELECT key, label, extra, x, y, region FROM {s}.force_map_point
                    WHERE entity = %s ORDER BY key""", (entity,))
    rows = cur.fetchall()
    index = {r[0]: i for i, r in enumerate(rows)}
    cur.execute(f"SELECT key_a, key_b FROM {s}.force_map_edge WHERE entity = %s", (entity,))
    pairs = [(index[a], index[b]) for a, b in cur.fetchall() if a in index and b in index]
    cur.execute(f"""SELECT region, size, name, x, y FROM {s}.force_map_region
                    WHERE entity = %s ORDER BY size DESC""", (entity,))
    regions = [{"region": r, "size": n, "name": name, "x": round(x, 1), "y": round(y, 1)}
               for r, n, name, x, y in cur.fetchall()]
    numeric = entity in ("book", "word")
    nodes = {
        "key": [int(r[0]) if numeric else r[0] for r in rows],
        "label": [r[1] for r in rows],
        "x": [round(r[3], 1) for r in rows],
        "y": [round(r[4], 1) for r in rows],
        "region": [r[5] for r in rows],
    }
    if entity == "book":
        nodes["extra"] = [r[2] for r in rows]
    body = {"entity": entity, "computed_at": computed_at.isoformat(), "nodes": nodes,
            "edges": {"a": [a for a, _ in pairs], "b": [b for _, b in pairs]}, "regions": regions}
    return json.dumps(body, separators=(",", ":")).encode()


@router.get("/api/browse/force-map")
def force_map(
    request: Request,
    entity: Literal["book", "author", "word"] = "book",
    _: dict = Depends(_main.require_viewer),
):
    """Every node (key, label, x, y, region; books also their author as
    `extra`, for the link), the drawn links, and the named regions of the
    latest run. 404 until `concordance force-map` has run for this entity."""
    s = _main.SCHEMA
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{s}.force_map_run",))
        run = None
        if cur.fetchone()[0] is not None:
            cur.execute(f"SELECT computed_at FROM {s}.force_map_run WHERE entity = %s", (entity,))
            run = cur.fetchone()
        if run is None:
            raise HTTPException(status_code=404, detail="This map hasn't been laid out yet.")
        etag = f'"force-{entity}-{run[0].timestamp():.0f}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        cached = _cache.get((s, entity))
        if cached is None or cached[0] != etag:
            cached = (etag, _build(cur, s, entity, run[0]))
            _cache[(s, entity)] = cached
    return Response(cached[1], media_type="application/json",
                    headers={"ETag": etag, "Cache-Control": "private, no-cache"})


@router.get("/api/browse/force-map/detail")
def force_map_detail(
    entity: Literal["book", "author", "word"],
    key: str,
    _: dict = Depends(_main.require_viewer),
):
    """A short summary of one map node, for the hover tooltip: a word's part
    of speech, definition, subject category and difficulty; a book's author,
    year, genres and vocabulary size; an author's books and vocabulary."""
    s = _main.SCHEMA
    with _main.get_conn() as conn, conn.cursor() as cur:
        if entity == "word":
            cur.execute(f"""SELECT w.lemma, w.part_of_speech, w.definition, d.difficulty,
                                   (SELECT c.name FROM {s}.word_category wc JOIN {s}.category c ON c.id = wc.category_id
                                    WHERE wc.word_id = w.id ORDER BY wc.is_primary DESC, wc.confidence DESC NULLS LAST
                                    LIMIT 1)
                            FROM {s}.word w LEFT JOIN {s}.word_difficulty d ON d.word_id = w.id
                            WHERE w.id = %s""", (int(key),))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="not found")
            lemma, pos, definition, difficulty, category = row
            return {"title": lemma, "subtitle": pos, "text": definition,
                    "facts": [f for f in (category, f"difficulty {round(difficulty)}" if difficulty is not None else None) if f]}
        if entity == "book":
            cur.execute(f"""SELECT b.title, b.author, b.publication_year, st.word_count,
                                   (SELECT string_agg(genre, ', ' ORDER BY genre) FROM {s}.book_genre g WHERE g.book_id = b.id)
                            FROM {s}.book b LEFT JOIN {s}.book_stats st ON st.book_id = b.id WHERE b.id = %s""",
                        (int(key),))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="not found")
            title, author, year, words, genres = row
            return {"title": title, "subtitle": author, "text": genres,
                    "facts": [f for f in (str(year) if year else None,
                                          f"{words:,} vocabulary words" if words else None) if f]}
        cur.execute(f"SELECT book_count, word_count FROM {s}.author_stats WHERE author = %s", (key,))
        row = cur.fetchone()
        books, words = row if row else (None, None)
        return {"title": key, "subtitle": None, "text": None,
                "facts": [f for f in (f"{books:,} book{'s' if books != 1 else ''}" if books else None,
                                      f"{words:,} vocabulary words" if words else None) if f]}
