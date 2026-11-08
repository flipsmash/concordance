"""/api/browse/clusters: a name for each cluster of the top-N clustering runs
(book_cluster / author_cluster and their _fame variants), so the map and
dendrogram can say what a cluster IS instead of colouring twelve anonymous
groups. Same idea as corpus_map's labels: a cluster's most distinctive
words -- the share of its members using a word x log(that share / the
corpus-wide share) -- with a minimum share so one member's oddities can't
name a group."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Literal

from fastapi import Depends
from webapp.backend import main as _main

from .common import router

_LABEL_WORDS = 4
_TTL = 3600
_cache: dict[tuple, tuple[float, dict]] = {}


def _min_share(size: int) -> float:
    """A label word must be used by at least 20% of a cluster's members, and
    by two of them when the cluster has more than one."""
    return max(0.2, 2 / size) if size > 1 else 1.0


@router.get("/api/browse/clusters")
def clusters(
    entity: Literal["book", "author"] = "book",
    scope: Literal["volume", "fame"] = "volume",
    _: dict = Depends(_main.require_viewer),
):
    """{clusters: [{cluster_id, size, label_words, members}]} for the latest
    run; members are book ids (or author names)."""
    s = _main.SCHEMA
    table = f"{entity}_cluster" + ("_fame" if scope == "fame" else "")
    key_col = "book_id" if entity == "book" else "author"
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT {key_col}, cluster_id, computed_at FROM {s}.{table}")
        rows = cur.fetchall()
        if not rows:
            return {"clusters": []}
        stamp = max(r[2] for r in rows)
        key = (s, entity, scope, stamp)
        hit = _cache.get(key)
        if hit and time.monotonic() - hit[0] < _TTL:
            return hit[1]

        cluster_of = {k: c for k, c, _ in rows}
        members: dict[int, list] = defaultdict(list)
        for k, c, _ in rows:
            members[c].append(k)

        # which words each member uses (active words only)
        if entity == "book":
            cur.execute(f"""SELECT wb.book_id, wb.word_id FROM {s}.word_book wb
                            JOIN {s}.word w ON w.id = wb.word_id AND w.active
                            WHERE wb.book_id = ANY(%s)""", (list(cluster_of),))
        else:
            cur.execute(f"""SELECT DISTINCT b.author, wb.word_id FROM {s}.word_book wb
                            JOIN {s}.word w ON w.id = wb.word_id AND w.active
                            JOIN {s}.book b ON b.id = wb.book_id
                            WHERE b.author = ANY(%s)""", (list(cluster_of),))
        users: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))   # cluster -> word -> members
        for k, wid in cur.fetchall():
            users[cluster_of[k]][wid] += 1
        words = list({w for u in users.values() for w in u})

        # corpus-wide share of books using each word
        cur.execute(f"SELECT count(*) FROM {s}.book")
        total = cur.fetchone()[0] or 1
        cur.execute(f"SELECT word_id, count(*) FROM {s}.word_book WHERE word_id = ANY(%s) GROUP BY word_id", (words,))
        share_all = {w: n / total for w, n in cur.fetchall()}

        out = []
        best_by_cluster: dict[int, list[int]] = {}
        for c, mem in members.items():
            size = len(mem)
            floor = _min_share(size)
            scored = []
            for w, n in users[c].items():
                sh = n / size
                g = share_all.get(w, 0)
                if sh >= floor and 0 < g <= 0.5:
                    scored.append((sh * math.log(sh / g), w))
            scored.sort(reverse=True)
            best_by_cluster[c] = [w for _, w in scored[:_LABEL_WORDS]]
        ids = sorted({w for ws in best_by_cluster.values() for w in ws})
        cur.execute(f"SELECT id, lemma FROM {s}.word WHERE id = ANY(%s)", (ids,))
        lemma = dict(cur.fetchall())
        for c in sorted(members):
            out.append({"cluster_id": c, "size": len(members[c]),
                        "label_words": [lemma[w] for w in best_by_cluster[c] if w in lemma],
                        "members": members[c]})
    result = {"clusters": out}
    _cache[key] = (time.monotonic(), result)
    return result
