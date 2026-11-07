"""/api/browse/origins: where the collection's words come from (see
concordance/origins.py and `concordance word-origins`), optionally within one
genre, plus the words behind any segment of the chart."""

from __future__ import annotations

from fastapi import Depends, Query
from webapp.backend import main as _main

from .common import router


def _scope(genre: str | None) -> tuple[str, list]:
    """SQL restricting word ids to active words (in books of `genre`)."""
    s = _main.SCHEMA
    if not genre:
        return f"SELECT id FROM {s}.word WHERE active", []
    return (f"""SELECT DISTINCT wb.word_id FROM {s}.word_book wb
                JOIN {s}.book_genre bg ON bg.book_id = wb.book_id
                JOIN {s}.word w ON w.id = wb.word_id AND w.active
                WHERE bg.genre = %s""", [genre])


@router.get("/api/browse/origins")
def origins_summary(genre: str | None = None, _: dict = Depends(_main.require_viewer)):
    """Counts by root family > route, and by family > root language, for every
    active word with a known origin (in `genre`, if given). `total_words` is
    every word in scope, known or not, so the page can state coverage."""
    s = _main.SCHEMA
    scope, params = _scope(genre)
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{s}.word_origin",))
        if cur.fetchone()[0] is None:
            return {"genre": genre, "total_words": 0, "known_words": 0, "families": []}
        cur.execute(f"SELECT count(*) FROM ({scope}) sc", params)
        total = cur.fetchone()[0]
        cur.execute(f"""SELECT o.family, o.route, count(*) FROM {s}.word_origin o
                        WHERE o.word_id IN ({scope}) GROUP BY 1, 2""", params)
        routes = cur.fetchall()
        cur.execute(f"""SELECT o.family, o.root_lang, o.root_lang_name, count(*) FROM {s}.word_origin o
                        WHERE o.word_id IN ({scope}) GROUP BY 1, 2, 3""", params)
        langs = cur.fetchall()
    fams: dict[str, dict] = {}
    for fam, route, n in routes:
        f = fams.setdefault(fam, {"family": fam, "count": 0, "routes": [], "languages": []})
        f["count"] += n
        f["routes"].append({"route": route, "count": n})
    for fam, code, name, n in langs:
        fams[fam]["languages"].append({"code": code, "name": name, "count": n})
    for f in fams.values():
        f["routes"].sort(key=lambda r: -r["count"])
        f["languages"].sort(key=lambda r: -r["count"])
    families = sorted(fams.values(), key=lambda f: -f["count"])
    return {"genre": genre, "total_words": total, "known_words": sum(f["count"] for f in families),
            "families": families}


@router.get("/api/browse/origins/words")
def origins_words(
    family: str,
    route: str | None = None,
    lang: str | None = None,
    genre: str | None = None,
    limit: int = Query(60, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: dict = Depends(_main.require_viewer),
):
    """The words in one segment (family, optionally narrowed to a route or a
    root language), the most widely read first so familiar examples lead."""
    s = _main.SCHEMA
    scope, params = _scope(genre)
    where = ["o.family = %s"]
    args: list = [family]
    if route:
        where.append("o.route = %s")
        args.append(route)
    if lang:
        where.append("o.root_lang = %s")
        args.append(lang)
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""SELECT count(*) FROM {s}.word_origin o
                        WHERE {' AND '.join(where)} AND o.word_id IN ({scope})""", args + params)
        total = cur.fetchone()[0]
        cur.execute(f"""SELECT w.id, w.lemma, left(coalesce(w.definition, ''), 140), o.root_lang_name, o.route,
                               (SELECT count(*) FROM {s}.word_book wb WHERE wb.word_id = w.id) AS books
                        FROM {s}.word_origin o JOIN {s}.word w ON w.id = o.word_id
                        WHERE {' AND '.join(where)} AND o.word_id IN ({scope})
                        ORDER BY books DESC, w.lemma LIMIT %s OFFSET %s""", args + params + [limit, offset])
        rows = cur.fetchall()
    return {"total": total, "words": [{"id": i, "lemma": l, "definition": d, "language": n, "route": r, "books": b}
                                      for i, l, d, n, r, b in rows]}
