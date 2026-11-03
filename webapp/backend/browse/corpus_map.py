"""/api/browse/corpus-map: the full-corpus map (every book or author), as
computed by `concordance corpus-map` (concordance/corpus_map.py).

Columnar on purpose -- parallel arrays, not one object per point: ~29k books
as objects is several MB of repeated keys. Revalidated by ETag (the run's
computed_at), so a browser downloads it again only after a new run."""

from __future__ import annotations

from typing import Literal

from fastapi import Depends, Request
from fastapi.responses import JSONResponse, Response
from webapp.backend import main as _main

from .common import router


@router.get("/api/browse/corpus-map")
def corpus_map(
    request: Request,
    entity: Literal["book", "author"] = "book",
    _: dict = Depends(_main.require_viewer),
):
    """Every point (key, label, x, y, cluster, subcluster, word_count,
    sparse; books also carry their author) and every cluster (top-level
    rows have subcluster -1) of the latest run, plus when it was computed.
    `computed_at` is null when the map has never been computed. A book
    added after the run is simply absent -- the page says so."""
    s = _main.SCHEMA
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{s}.corpus_map_run",))
        if cur.fetchone()[0] is None:
            return JSONResponse({"entity": entity, "computed_at": None, "n": 0, "points": None, "clusters": []})
        cur.execute(f"SELECT n, computed_at, params FROM {s}.corpus_map_run WHERE entity = %s", (entity,))
        run = cur.fetchone()
        if run is None:
            return JSONResponse({"entity": entity, "computed_at": None, "n": 0, "points": None, "clusters": []})
        etag = f'"{entity}-{run[1].timestamp():.0f}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        if entity == "book":
            cur.execute(f"""SELECT p.key, p.label, coalesce(b.author, ''), p.x, p.y, p.cluster, p.subcluster,
                                   p.word_count, p.sparse
                            FROM {s}.corpus_map_point p LEFT JOIN {s}.book b ON b.id::text = p.key
                            WHERE p.entity = 'book' ORDER BY p.cluster, p.subcluster, p.key""")
        else:
            cur.execute(f"""SELECT key, label, '', x, y, cluster, subcluster, word_count, sparse
                            FROM {s}.corpus_map_point WHERE entity = 'author'
                            ORDER BY cluster, subcluster, key""")
        rows = cur.fetchall()
        cur.execute(f"""SELECT cluster, subcluster, size, x, y, label_words FROM {s}.corpus_map_cluster
                        WHERE entity = %s ORDER BY cluster, subcluster""", (entity,))
        clusters = [{"cluster": c, "subcluster": sc, "size": n, "x": round(x, 1), "y": round(y, 1),
                     "label_words": lw} for c, sc, n, x, y, lw in cur.fetchall()]
    cols = list(zip(*rows)) if rows else [()] * 9
    points = {
        "key": [int(k) if entity == "book" else k for k in cols[0]],
        "label": list(cols[1]),
        "x": [round(v, 1) for v in cols[3]],
        "y": [round(v, 1) for v in cols[4]],
        "cluster": list(cols[5]),
        "subcluster": list(cols[6]),
        "word_count": list(cols[7]),
        "sparse": [int(v) for v in cols[8]],
    }
    if entity == "book":
        points["author"] = list(cols[2])
    body = {"entity": entity, "computed_at": run[1].isoformat(), "n": run[0], "params": run[2],
            "points": points, "clusters": clusters}
    return JSONResponse(body, headers={"ETag": etag, "Cache-Control": "private, no-cache"})
