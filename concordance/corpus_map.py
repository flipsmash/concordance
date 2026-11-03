"""`concordance corpus-map`: every book and every author on one 2-D map.

The relatedness pages' map/matrix/dendrogram show the top 200 only: their
Ward linkage and n x n MDS stop being tractable (and legible) long before
real corpus scale. This places EVERYTHING instead, for a zoomable map:

  vectors   the same measure as book-similarity / author-similarity: ACTIVE
            words, IDF-weighted (author-document frequency for authors),
            words in more than half of all books/authors left out, rows
            L2-normalised -- so the map and the neighbour lists measure
            the same thing
  layout    TruncatedSVD to ~100 dimensions, then UMAP to 2-D (cosine, fixed
            seed, so a re-run over the same data gives the same map)
  clusters  KMeans on a separate 10-D UMAP embedding, two levels. KMeans,
            not HDBSCAN: the corpus is one continuous spread of vocabulary,
            not islands, and HDBSCAN either left a third to a half of all
            books unclustered or merged 27k of 29k into one blob (measured
            2026-11-03). Cluster counts scale with n (about sqrt(n) / 4),
            never a count tuned to today's corpus.
  labels    each cluster's most distinctive words: share of its members
            using the word x log(that share / the corpus-wide share), with a
            minimum share so one book's oddities can't label a cluster

Positions are approximate (2-D can't keep every neighbourhood); the exact
neighbour lists stay book_similarity / author_similarity. Not part of
`maintain`: a full recompute, run when a fresh map is wanted.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .db import _safe_schema
from .db.books import PLACEHOLDER_AUTHORS

MAX_DF_FRACTION = 0.5       # same cut as book-similarity / author-similarity
SVD_DIMS = 100
N_NEIGHBORS = 15
SEED = 0
LABEL_WORDS = 8
LABEL_MIN_SHARE = 0.05      # a label word is used by at least 5% of the cluster
SPARSE_WORDS = 20           # fewer qualifying words than this: placed, but flagged
COORD_SCALE = 1000.0        # coordinates stored in [0, COORD_SCALE]


def n_clusters(n: int) -> int:
    """Top-level clusters for n points (and sub-clusters for a cluster of n):
    about sqrt(n) / 4 -- ~43 for 29k books, ~7 inside an 800-book cluster."""
    return max(1, min(n, round(math.sqrt(n) / 4)))


def _rows(conn, s: str, entity: str):
    """(keys, labels, pairs) -- pairs are (row key, word id), each word once
    per row. Authors exclude the placeholder names, as author-similarity does."""
    with conn.cursor() as cur:
        if entity == "book":
            cur.execute(f"""SELECT wb.book_id, wb.word_id FROM {s}.word_book wb
                            JOIN {s}.word w ON w.id = wb.word_id WHERE w.active""")
            pairs = cur.fetchall()
            cur.execute(f"SELECT id, title FROM {s}.book")
            names = {i: t for i, t in cur.fetchall()}
        else:
            cur.execute(f"""SELECT DISTINCT b.author, wb.word_id FROM {s}.word_book wb
                            JOIN {s}.word w ON w.id = wb.word_id
                            JOIN {s}.book b ON b.id = wb.book_id
                            WHERE w.active AND b.author IS NOT NULL AND b.author <> ''
                              AND NOT (b.author = ANY(%s))""", (list(PLACEHOLDER_AUTHORS),))
            pairs = cur.fetchall()
            names = None
    conn.commit()
    return pairs, names


def build_matrix(pairs):
    """(row keys, word ids, L2-normalised IDF matrix, qualifying words per row,
    binary row x word matrix) from (row key, word id) pairs."""
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import normalize
    keys, ri = np.unique(np.array([p[0] for p in pairs], dtype=object), return_inverse=True)
    words, wi = np.unique(np.array([p[1] for p in pairs], dtype=np.int64), return_inverse=True)
    n = len(keys)
    df = np.bincount(wi, minlength=len(words))
    keep = df <= MAX_DF_FRACTION * n
    idf = np.log(n / df)
    m = keep[wi]
    X = csr_matrix((idf[wi[m]], (ri[m], wi[m])), shape=(n, len(words)))
    X.sum_duplicates()
    B = csr_matrix((np.ones(m.sum(), dtype=np.float32), (ri[m], wi[m])), shape=(n, len(words)))
    B.sum_duplicates()
    B.data[:] = 1
    return list(keys), words, normalize(X), np.diff(X.indptr), B


def embed(X, seed: int = SEED):
    """(2-D layout, 10-D clustering embedding)."""
    import umap
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    n, v = X.shape
    dims = max(2, min(SVD_DIMS, n - 1, v - 1))
    Z = normalize(TruncatedSVD(dims, random_state=seed).fit_transform(X))
    nn = max(2, min(N_NEIGHBORS, n - 1))
    U2 = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.05, metric="cosine",
                   random_state=seed).fit_transform(Z)
    U10 = umap.UMAP(n_components=min(10, n - 2), n_neighbors=nn, min_dist=0.0, metric="cosine",
                    random_state=seed).fit_transform(Z)
    return U2, U10


def cluster(U10, seed: int = SEED):
    """(top-level label per point, sub-cluster label per point)."""
    from sklearn.cluster import KMeans
    n = len(U10)
    top = KMeans(n_clusters(n), n_init=4, random_state=seed).fit_predict(U10)
    sub = np.zeros(n, dtype=int)
    for c in np.unique(top):
        rows = np.where(top == c)[0]
        k = n_clusters(len(rows))
        if k > 1:
            sub[rows] = KMeans(k, n_init=4, random_state=seed).fit_predict(U10[rows])
    return top, sub


def label_words(B, members, share_all, lemmas) -> list[str]:
    """A cluster's most distinctive words (see the module docstring)."""
    share = np.asarray(B[members].sum(0)).ravel() / max(1, len(members))
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(share >= LABEL_MIN_SHARE, share * np.log(share / share_all), 0.0)
    top = [i for i in np.argsort(-score)[:LABEL_WORDS] if score[i] > 0]
    return [lemmas[i] for i in top]


def compute(conn, schema: str, entity: str, seed: int = SEED, progress=print) -> dict:
    """Compute and store one entity's map ('book' or 'author')."""
    t0 = time.time()
    s = _safe_schema(schema)
    pairs, names = _rows(conn, s, entity)
    if len({p[0] for p in pairs}) < 4:
        return {"entity": entity, "n": 0}
    keys, words, X, nwords, B = build_matrix(pairs)
    # A row whose words are ALL too common has an empty vector: no position
    # to give it (and cosine is undefined). Left off the map, and counted.
    has = nwords > 0
    empty = int((~has).sum())
    if empty:
        keys = [k for k, h in zip(keys, has) if h]
        X, B, nwords = X[has], B[has], nwords[has]
    progress(f"  {entity}s: {len(keys):,} x {X.shape[1]:,} qualifying words"
             + (f" ({empty} with none, left off the map)" if empty else ""))
    U2, U10 = embed(X, seed)
    progress(f"  layout done ({time.time() - t0:.0f}s)")
    top, sub = cluster(U10, seed)

    lo, hi = U2.min(0), U2.max(0)
    xy = (U2 - lo) / np.where(hi > lo, hi - lo, 1) * COORD_SCALE
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma FROM {s}.word WHERE id = ANY(%s)", (words.tolist(),))
        lemma_by_id = dict(cur.fetchall())
    lemmas = [lemma_by_id.get(int(w), "") for w in words]
    share_all = np.asarray(B.sum(0)).ravel() / len(keys)

    clusters = []
    for c in np.unique(top):
        rows = np.where(top == c)[0]
        clusters.append((int(c), -1, len(rows), *xy[rows].mean(0), label_words(B, rows, share_all, lemmas)))
        for sc in np.unique(sub[rows]):
            srows = rows[sub[rows] == sc]
            clusters.append((int(c), int(sc), len(srows), *xy[srows].mean(0),
                             label_words(B, srows, share_all, lemmas)))

    seconds = time.time() - t0
    params = {"svd_dims": SVD_DIMS, "n_neighbors": N_NEIGHBORS, "seed": seed,
              "max_df_fraction": MAX_DF_FRACTION, "clusters": int(top.max()) + 1,
              "label_min_share": LABEL_MIN_SHARE, "sparse_words": SPARSE_WORDS}
    from psycopg.types.json import Json
    with conn.cursor() as cur:
        # One transaction: readers keep the previous map until this commits.
        cur.execute(f"DELETE FROM {s}.corpus_map_point WHERE entity = %s", (entity,))
        cur.execute(f"DELETE FROM {s}.corpus_map_cluster WHERE entity = %s", (entity,))
        with cur.copy(f"COPY {s}.corpus_map_point (entity, key, label, x, y, cluster, subcluster, "
                      f"word_count, sparse) FROM STDIN") as copy:
            for i, k in enumerate(keys):
                label = (names or {}).get(k, k) if entity == "book" else k
                copy.write_row((entity, str(k), label or str(k), float(xy[i, 0]), float(xy[i, 1]),
                                int(top[i]), int(sub[i]), int(nwords[i]), bool(nwords[i] < SPARSE_WORDS)))
        with cur.copy(f"COPY {s}.corpus_map_cluster (entity, cluster, subcluster, size, x, y, label_words) "
                      f"FROM STDIN") as copy:
            for c, sc, size, x, y, lw in clusters:
                copy.write_row((entity, c, sc, size, float(x), float(y), lw))
        cur.execute(f"""INSERT INTO {s}.corpus_map_run (entity, n, params, seconds, computed_at)
                        VALUES (%s, %s, %s, %s, now())
                        ON CONFLICT (entity) DO UPDATE SET n = EXCLUDED.n, params = EXCLUDED.params,
                            seconds = EXCLUDED.seconds, computed_at = now()""",
                    (entity, len(keys), Json(params), seconds))
    conn.commit()
    return {"entity": entity, "n": len(keys), "empty": empty, "clusters": params["clusters"],
            "subclusters": sum(1 for c in clusters if c[1] >= 0),
            "sparse": int((nwords < SPARSE_WORDS).sum()), "seconds": round(seconds, 1)}


def neighbour_agreement(conn, schema: str, entity: str = "book") -> dict:
    """How well the map agrees with the exact neighbour lists: the share of
    each point's stored top-k neighbours that land in its own top-level
    cluster, against what random clusters of the same sizes would give."""
    s = _safe_schema(schema)
    table, a_col, b_col = (("book_similarity", "book_a_id::text", "book_b_id::text") if entity == "book"
                           else ("author_similarity", "author_a", "author_b"))
    with conn.cursor() as cur:
        cur.execute(f"""SELECT avg((pa.cluster = pb.cluster)::int)
                        FROM {s}.{table} t
                        JOIN {s}.corpus_map_point pa ON pa.entity = %s AND pa.key = t.{a_col}
                        JOIN {s}.corpus_map_point pb ON pb.entity = %s AND pb.key = t.{b_col}""",
                    (entity, entity))
        same = cur.fetchone()[0]
        cur.execute(f"""SELECT sum(power(size::float / tot, 2)) FROM {s}.corpus_map_cluster,
                        (SELECT sum(size) tot FROM {s}.corpus_map_cluster WHERE entity = %s AND subcluster = -1) t
                        WHERE entity = %s AND subcluster = -1""", (entity, entity))
        chance = cur.fetchone()[0]
    conn.commit()
    return {"same_cluster": float(same or 0), "chance": float(chance or 0)}
