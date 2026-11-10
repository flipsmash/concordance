"""Force-directed maps of the whole collection: every book, every author,
every word, placed by its nearest neighbours and drawn by the "Force map"
views (precomputed here, explored by zooming and dragging in the browser).

The graph:
  - books / authors: each one's strongest links from book_similarity /
    author_similarity (shared rare vocabulary);
  - words: each word's nearest neighbours by definition meaning (cosine on
    the 384-d definition embeddings, computed in chunks with numpy).
Each node keeps its top K links; a pair counts once, at its stronger weight.

The layout is ForceAtlas2 in LinLog mode (graphology's implementation, run
with Node: webapp/frontend/scripts/force_layout.mjs) -- LinLog pulls
communities apart into separate regions. Two passes: a settling pass with
moderate gravity, then a long one with almost none so clusters open up.
Recomputing starts from the last run's positions (else, for books and
authors, from the UMAP corpus map), so the picture stays recognizable from
one run to the next instead of being redrawn from scratch.

Regions: Leiden communities on the same graph (igraph). Books and authors
name theirs by distinctive vocabulary, as the corpus map does; words by
their best-connected members.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .db import _safe_schema
from .db.books import PLACEHOLDER_AUTHORS

TOP_K = 8                  # links per node in the layout graph
SHOWN_K = {"book": 4, "author": 4, "word": 3}   # links per node kept for drawing
SEED = 42
_SCRIPT = Path(__file__).resolve().parents[1] / "webapp" / "frontend" / "scripts" / "force_layout.mjs"
_PASSES = [
    (1500, {"linLogMode": True, "barnesHutOptimize": True, "barnesHutTheta": 1.0,
            "scalingRatio": 2, "gravity": 0.5, "slowDown": 1}),
    (2000, {"linLogMode": True, "barnesHutOptimize": True, "barnesHutTheta": 1.0,
            "scalingRatio": 1, "gravity": 0.01, "edgeWeightInfluence": 1.5, "slowDown": 1}),
]


def node_binary() -> str:
    """Node, which the web service's own PATH may not have (it's nvm's)."""
    found = shutil.which("node")
    if found:
        return found
    nvm = sorted(Path.home().glob(".nvm/versions/node/*/bin/node"))
    if nvm:
        return str(nvm[-1])
    raise RuntimeError("force-map needs Node (for graphology's ForceAtlas2); none found")


# --- the graphs ---------------------------------------------------------------

def _top_k(pairs, k: int) -> dict[tuple, float]:
    """Each node's k strongest links; a pair once, at its stronger weight."""
    per: dict = defaultdict(list)
    for a, b, w in pairs:
        per[a].append((w, b))
        per[b].append((w, a))
    edges: dict[tuple, float] = {}
    for a, lst in per.items():
        for w, b in sorted(lst, key=lambda t: -t[0])[:k]:
            key = (a, b) if str(a) < str(b) else (b, a)
            edges[key] = max(edges.get(key, 0.0), float(w))
    return edges


def graph(conn, schema: str, entity: str, k: int = TOP_K) -> tuple[list, list[str], dict[tuple, float]]:
    """(node keys, labels, {(key a, key b): weight}). A book's label is its
    title; the author (needed for its page's URL) comes from book_authors."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        if entity == "book":
            cur.execute(f"SELECT book_a_id, book_b_id, score FROM {s}.book_similarity")
            edges = _top_k(cur.fetchall(), k)
            cur.execute(f"SELECT id, title FROM {s}.book")
            names = dict(cur.fetchall())
        elif entity == "author":
            cur.execute(f"""SELECT author_a, author_b, score FROM {s}.author_similarity
                            WHERE NOT (author_a = ANY(%s)) AND NOT (author_b = ANY(%s))""",
                        (list(PLACEHOLDER_AUTHORS), list(PLACEHOLDER_AUTHORS)))
            edges = _top_k(cur.fetchall(), k)
            names = None
        else:
            return _word_graph(conn, s, k)
    keys = sorted({x for e in edges for x in e}, key=str)
    labels = [names.get(key, str(key)) if names else str(key) for key in keys]
    return keys, labels, edges


_WORD_VECTORS: dict = {}   # keys -> normalized vectors, reused by the UMAP start


def _word_graph(conn, s: str, k: int):
    """Every active embedded word's k nearest by definition meaning."""
    with conn.cursor() as cur:
        cur.execute(f"""SELECT w.id, w.lemma, e.definition_vector::text FROM {s}.word w
                        JOIN {s}.word_embedding e ON e.word_id = w.id
                        WHERE w.active AND e.definition_vector IS NOT NULL ORDER BY w.id""")
        rows = cur.fetchall()
    keys = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    V = np.array([np.fromstring(r[2].strip("[]"), sep=",", dtype=np.float32) for r in rows])
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    _WORD_VECTORS["keys"], _WORD_VECTORS["V"] = keys, V
    pairs = []
    for start in range(0, len(V), 2000):
        sims = V[start:start + 2000] @ V.T
        for r in range(sims.shape[0]):
            sims[r, start + r] = -1.0                    # not its own neighbour
        nn = np.argpartition(-sims, k, axis=1)[:, :k]
        for r, cols in enumerate(nn):
            for c in cols:
                pairs.append((keys[start + r], keys[c], max(float(sims[r, c]), 0.0)))
    return keys, labels, _top_k(pairs, k)


# --- the layout -----------------------------------------------------------------

def _start_positions(conn, s: str, entity: str, keys: list, fresh: bool = False) -> np.ndarray:
    """Last run's positions (unless `fresh`), else the UMAP corpus map's
    (books/authors) or a UMAP projection of the meaning embeddings (words) --
    ForceAtlas2 refines a good start far better than a random one at this
    size. Nodes new since the last run start at random."""
    rng = np.random.default_rng(SEED)
    pos = rng.normal(scale=100.0, size=(len(keys), 2))
    with conn.cursor() as cur:
        prev = {}
        if not fresh:
            cur.execute(f"SELECT key, x, y FROM {s}.force_map_point WHERE entity = %s", (entity,))
            prev = {k: (x, y) for k, x, y in cur.fetchall()}
        if not prev and entity == "word" and _WORD_VECTORS.get("keys") == keys:
            import umap
            U = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine",
                          random_state=SEED).fit_transform(_WORD_VECTORS["V"])
            return U * 100
        if not prev and entity in ("book", "author"):
            cur.execute(f"SELECT key, x, y FROM {s}.corpus_map_point WHERE entity = %s", (entity,))
            prev = {k: (x * 100, y * 100) for k, x, y in cur.fetchall()}
    for i, key in enumerate(keys):
        if str(key) in prev:
            pos[i] = prev[str(key)]
    return pos


def layout(n: int, edge_index: list[tuple[int, int, float]], start: np.ndarray, progress=print) -> np.ndarray:
    weights = np.array([w for _, _, w in edge_index], dtype=float)
    scale = weights.mean() if len(weights) and weights.mean() > 0 else 1.0
    pos = start
    for iterations, settings in _PASSES:
        payload = {"nodes": [{"x": float(x), "y": float(y)} for x, y in pos],
                   "edges": [[a, b, w / scale] for a, b, w in edge_index],
                   "iterations": iterations, "settings": settings}
        proc = subprocess.run([node_binary(), str(_SCRIPT)], input=json.dumps(payload),
                              capture_output=True, text=True, cwd=_SCRIPT.parents[1],
                              env={**os.environ, "NODE_OPTIONS": "--max-old-space-size=8192"})
        if proc.returncode != 0:
            raise RuntimeError(f"force layout failed: {proc.stderr[-500:]}")
        progress(proc.stderr.strip())
        out = json.loads(proc.stdout)
        pos = np.c_[out["x"], out["y"]]
    return pos


# --- regions ---------------------------------------------------------------------

def communities(n: int, edge_index) -> np.ndarray:
    import igraph as ig
    g = ig.Graph(n=n, edges=[(a, b) for a, b, _ in edge_index])
    g.es["weight"] = [w for _, _, w in edge_index]
    return np.array(g.community_leiden(objective_function="modularity", weights="weight",
                                       n_iterations=5, initial_membership=None).membership)


def _region_names(conn, s: str, entity: str, keys: list, labels: list[str], comm: np.ndarray,
                  edge_index) -> dict[int, str]:
    """Up to the 40 biggest regions get a name."""
    biggest = [c for c, _ in Counter(comm.tolist()).most_common(40)]
    names: dict[int, str] = {}
    if entity == "word":
        # the region's most common subject category, then its best-connected words
        degree = np.zeros(len(keys))
        for a, b, w in edge_index:
            degree[a] += w
            degree[b] += w
        with conn.cursor() as cur:
            cur.execute(f"""SELECT DISTINCT ON (wc.word_id) wc.word_id, c.name FROM {s}.word_category wc
                            JOIN {s}.category c ON c.id = wc.category_id
                            ORDER BY wc.word_id, wc.is_primary DESC, wc.confidence DESC NULLS LAST""")
            category = dict(cur.fetchall())
        for c in biggest:
            members = np.where(comm == c)[0]
            top = members[np.argsort(-degree[members])[:3]]
            cats = Counter(category.get(keys[i]) for i in members if category.get(keys[i]))
            words = ", ".join(labels[i] for i in top)
            names[c] = f"{cats.most_common(1)[0][0].capitalize()}: {words}" if cats else words
        return names
    from . import corpus_map
    pairs, _ = corpus_map._rows(conn, s, entity)
    if not pairs:
        return names
    row_keys, words, _X, _q, B = corpus_map.build_matrix(pairs)
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma FROM {s}.word WHERE id = ANY(%s)", ([int(w) for w in words],))
        lemma = dict(cur.fetchall())
    lemmas = [lemma.get(int(w), "") for w in words]
    row_of = {str(k): i for i, k in enumerate(row_keys)}
    share_all = np.asarray(B.sum(0)).ravel() / max(1, B.shape[0])
    for c in biggest:
        rows = [row_of[str(keys[i])] for i in np.where(comm == c)[0] if str(keys[i]) in row_of]
        if rows:
            names[c] = ", ".join(corpus_map.label_words(B, rows, share_all, lemmas)[:3])
    return names


def book_authors(cur, s: str) -> dict:
    cur.execute(f"SELECT id, author FROM {s}.book")
    return dict(cur.fetchall())


# --- the whole step -----------------------------------------------------------------

def compute(conn, schema: str, entity: str, progress=print, fresh: bool = False) -> dict:
    """Build, lay out, name and store one entity's force map; returns counts."""
    s = _safe_schema(schema)
    progress(f"{entity}: building the graph…")
    keys, labels, edges = graph(conn, schema, entity)
    index = {k: i for i, k in enumerate(keys)}
    edge_index = [(index[a], index[b], w) for (a, b), w in edges.items()]
    progress(f"{entity}: {len(keys):,} nodes, {len(edge_index):,} links; laying out…")
    pos = layout(len(keys), edge_index, _start_positions(conn, s, entity, keys, fresh), progress)
    comm = communities(len(keys), edge_index)
    names = _region_names(conn, s, entity, keys, labels, comm, edge_index)
    shown = {(a, b): w for (a, b), w in _top_k([(a, b, w) for (a, b), w in edges.items()], SHOWN_K[entity]).items()}

    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {s}.force_map_point WHERE entity = %s", (entity,))
        cur.execute(f"DELETE FROM {s}.force_map_edge WHERE entity = %s", (entity,))
        cur.execute(f"DELETE FROM {s}.force_map_region WHERE entity = %s", (entity,))
        extra = book_authors(cur, s) if entity == "book" else {}
        cur.executemany(f"""INSERT INTO {s}.force_map_point (entity, key, label, extra, x, y, region)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                        [(entity, str(k), labels[i], extra.get(k), float(pos[i, 0]), float(pos[i, 1]),
                          int(comm[i])) for i, k in enumerate(keys)])
        cur.executemany(f"""INSERT INTO {s}.force_map_edge (entity, key_a, key_b, weight)
                            VALUES (%s, %s, %s, %s)""",
                        [(entity, str(a), str(b), float(w)) for (a, b), w in shown.items()])
        sizes = Counter(comm.tolist())
        cur.executemany(f"""INSERT INTO {s}.force_map_region (entity, region, size, name, x, y)
                            VALUES (%s, %s, %s, %s, %s, %s)""",
                        [(entity, c, sizes[c], names.get(c),
                          float(np.median(pos[comm == c, 0])), float(np.median(pos[comm == c, 1])))
                         for c in sizes])
        cur.execute(f"""INSERT INTO {s}.force_map_run (entity, nodes, edges, computed_at)
                        VALUES (%s, %s, %s, now())
                        ON CONFLICT (entity) DO UPDATE SET nodes = EXCLUDED.nodes, edges = EXCLUDED.edges,
                            computed_at = now()""", (entity, len(keys), len(shown)))
    conn.commit()
    return {"nodes": len(keys), "edges": len(edge_index), "shown_edges": len(shown),
            "regions": len(set(comm.tolist())), "named": len(names)}
