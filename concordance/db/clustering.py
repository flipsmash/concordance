"""Hierarchical clustering of authors and books for the browse visualizations."""

from __future__ import annotations

from .books import PLACEHOLDER_AUTHORS
from .core import DEFAULT_SCHEMA, _safe_schema


def _linkage_to_tree(Z, leaf_data: list[dict]) -> dict:
    """scipy.cluster.hierarchy.linkage's output (an (n-1)x4 array: each row
    [left, right, distance, size], where left/right index either an
    original leaf (0..n-1) or a previously-built internal node (n..)) into
    a nested, JSON-serializable tree for the frontend dendrogram -- one
    bottom-up pass, no recursion needed since Z is already in merge order.

    leaf_data[i] is spread directly into leaf i's dict (e.g. {"author": name}
    for compute_author_clustering, {"id", "title", "author"} for
    compute_book_clustering) -- generic over what identifies a leaf rather
    than assuming a single string label, since a book (unlike an author)
    needs more than one field to be both displayable and navigable."""
    n = len(leaf_data)
    nodes: dict[int, dict] = {i: {**leaf_data[i], "size": 1} for i in range(n)}
    for i, (a, b, dist, size) in enumerate(Z):
        nodes[n + i] = {
            "left": nodes[int(a)],
            "right": nodes[int(b)],
            "distance": float(dist),
            "size": int(size),
        }
    return nodes[n + len(Z) - 1]


def compute_author_clustering(conn, schema: str = DEFAULT_SCHEMA, *, top_n: int = 200,
                              max_df_fraction: float = 0.5, n_clusters: int = 12,
                              min_fame: float | None = None) -> dict:
    """`concordance author-clustering` / `maintain`'s clustering step: the
    data behind the cluster map, similarity matrix, and dendrogram views for
    the top `top_n` authors by book count.

    min_fame switches the SELECTION criterion (and the destination tables)
    entirely: instead of the top_n most-represented authors, every author
    with author_fame.fame_score >= min_fame is included (top_n is ignored
    in this mode -- there's no volume cap, since fame-scoring itself is a
    slow, deliberately-manual process, see fame.py's own docstring, so this
    set stays naturally small without an artificial limit). Writes to
    author_cluster_fame/author_cluster_fame_run instead of author_cluster/
    author_cluster_run, a second, independent lens on the corpus ("most
    historically important" vs "most-represented") rather than a
    replacement -- both runs coexist. Deliberately NOT wired into `maintain`
    (unlike the top_n-by-volume run): fame scores change rarely and only via
    a separate manually-run command (book-fame/author-fame), so recomputing
    this on every maintain pass would be pure churn -- rerun it by hand
    (`concordance author-clustering --min-fame 8`) after a fame-scoring run
    actually changes the qualifying set.

    Reuses the exact corpus-wide author-df IDF setup compute_author_similarity
    uses (same n_authors/df/max_df_fraction computation over ALL authors,
    not just the top_n) -- so "the why" (author_similarity's scores) and
    "the map" (cluster positions here) share one consistent notion of
    similarity; only WHICH authors enter the pairwise/clustering step is
    restricted to top_n, not how a shared word is weighted.

    PLACEHOLDER_AUTHORS ("Various", "Unknown Author", ...) are filtered in
    the SAME WHERE clause as the top_n ORDER BY/LIMIT, not afterward --
    they dwarf every real author by book count (Various alone: 1193 books
    vs. ~125 for the top real author), so a post-hoc filter would silently
    burn top_n slots on aggregation labels instead of real authors.

    Distance is `sqrt(2 * (1 - cosine))`, a proper Euclidean distance for
    L2-normalized vectors -- not raw `1 - cosine`, which fails the triangle
    inequality and produces a non-PSD Gram matrix (forcing lossy eigenvalue
    clipping in the MDS step below). This one distance definition feeds
    both `ward` linkage (which assumes squared-Euclidean input -- not valid
    for raw cosine distance) and classical MDS, rather than being decided
    independently in two places.

    n_clusters=12 (via `fcluster(..., criterion='maxclust')`) is a starting
    default, not a permanent one -- validate cluster-size distribution
    against the real corpus (not all-one-cluster, not all-singletons)
    before treating it as final.

    Classical (Torgerson) MDS is computed directly via a single
    numpy.linalg.eigh on the double-centered squared-distance Gram matrix
    (no sklearn) -- fast and deterministic at this scale, unlike sklearn's
    default iterative SMACOF. eigh's eigenvector sign is otherwise
    arbitrary and can flip between runs on near-identical input (the same
    instability class as the force-graph bugs already found and fixed
    elsewhere in this project), so each axis's sign is pinned deterministically.
    Authors landing on the exact same point (identical qualifying word
    sets) get a small, deterministic (name-hash-seeded) jitter so they stay
    individually clickable.

    Writes author_cluster and the singleton author_cluster_run in one
    transaction at the end -- no partial/interleaved commits (unlike
    compute_book_similarity's every-200-rows batching): top_n=200 is small
    enough that one clean transaction is trivial, and a partial write here
    would be worse than there, since the map/matrix/dendrogram must never
    disagree with each other -- they all derive from one computation pass."""
    import hashlib
    import math

    import numpy as np
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform
    from scipy.sparse import csr_matrix

    s = _safe_schema(schema)
    placeholders = list(PLACEHOLDER_AUTHORS)
    cluster_table = "author_cluster_fame" if min_fame is not None else "author_cluster"
    cluster_run_table = "author_cluster_fame_run" if min_fame is not None else "author_cluster_run"

    with conn.cursor() as cur:
        cur.execute(f"""SELECT count(DISTINCT b.author) FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        JOIN {s}.book b ON b.id = wb.book_id
                        WHERE w.active AND b.author IS NOT NULL AND b.author <> ''
                          AND NOT (b.author = ANY(%s))""", (placeholders,))
        n_authors = cur.fetchone()[0]
        if n_authors < 3:
            conn.commit()  # see compute_book_similarity's own early-return commit note
            return {"authors": 0, "clusters": 0}

        cur.execute(f"""SELECT wb.word_id, count(DISTINCT b.author) AS df
                        FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        JOIN {s}.book b ON b.id = wb.book_id
                        WHERE w.active AND b.author IS NOT NULL AND b.author <> ''
                          AND NOT (b.author = ANY(%s))
                        GROUP BY wb.word_id""", (placeholders,))
        max_df = max_df_fraction * n_authors
        idf = {wid: math.log(n_authors / df) for wid, df in cur.fetchall() if df <= max_df}
        if not idf:
            conn.commit()
            return {"authors": 0, "clusters": 0}

        if min_fame is not None:
            cur.execute(f"""SELECT b.author, count(DISTINCT b.id) AS book_count
                            FROM {s}.book b
                            JOIN {s}.author_fame af ON af.author = b.author
                            WHERE b.author IS NOT NULL AND b.author <> ''
                              AND NOT (b.author = ANY(%s))
                              AND af.fame_score >= %s
                            GROUP BY b.author, af.fame_score
                            ORDER BY af.fame_score DESC, book_count DESC""", (placeholders, min_fame))
        else:
            cur.execute(f"""SELECT b.author, count(DISTINCT b.id) AS book_count
                            FROM {s}.book b
                            WHERE b.author IS NOT NULL AND b.author <> ''
                              AND NOT (b.author = ANY(%s))
                            GROUP BY b.author
                            ORDER BY book_count DESC
                            LIMIT %s""", (placeholders, top_n))
        top_authors = cur.fetchall()
        top_author_names = [r[0] for r in top_authors]
        book_count_by_author = {r[0]: r[1] for r in top_authors}

        if len(top_author_names) < 3:
            conn.commit()
            return {"authors": 0, "clusters": 0}

        cur.execute(f"""SELECT DISTINCT wb.word_id, b.author FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        JOIN {s}.book b ON b.id = wb.book_id
                        WHERE w.active AND b.author = ANY(%s)
                          AND wb.word_id = ANY(%s)""",
                    (top_author_names, list(idf.keys())))
        rows = cur.fetchall()

    n = len(top_author_names)
    author_index = {a: i for i, a in enumerate(top_author_names)}
    word_index: dict[int, int] = {}
    row_idx, col_idx, weighted_data = [], [], []
    for wid, author in rows:
        j = word_index.setdefault(wid, len(word_index))
        row_idx.append(author_index[author])
        col_idx.append(j)
        weighted_data.append(idf[wid])

    matrix = csr_matrix((weighted_data, (row_idx, col_idx)), shape=(n, len(word_index)))
    binary = csr_matrix(([1] * len(row_idx), (row_idx, col_idx)), shape=(n, len(word_index)))
    shared_counts = (binary @ binary.T).toarray().astype(int)

    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1))).reshape(-1)
    norms[norms == 0] = 1.0  # an author with zero qualifying words (all excluded by max_df) -- avoid /0
    normalized = matrix.multiply(1 / norms[:, None]).tocsr()
    cosine = (normalized @ normalized.T).toarray()
    np.clip(cosine, -1.0, 1.0, out=cosine)  # guard float round-off past 1 on a self-pair

    dist = np.sqrt(np.clip(2 * (1 - cosine), 0, None))
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    tree = linkage(condensed, method="ward", optimal_ordering=True)
    leaf_order_idx = leaves_list(tree)
    leaf_order = [top_author_names[i] for i in leaf_order_idx]
    cluster_ids = fcluster(tree, t=n_clusters, criterion="maxclust")

    # Classical (Torgerson) MDS: double-center the squared-distance matrix,
    # take the top-2 eigenvectors of the resulting Gram matrix.
    d2 = dist ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ d2 @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    top2 = np.argsort(eigvals)[::-1][:2]
    coords = eigvecs[:, top2] * np.sqrt(np.clip(eigvals[top2], 0, None))

    for axis in range(coords.shape[1]):
        col = coords[:, axis]
        if col[np.argmax(np.abs(col))] < 0:
            coords[:, axis] = -col

    seen_points: dict[tuple, list[int]] = {}
    for i in range(n):
        key = (round(float(coords[i, 0]), 6), round(float(coords[i, 1]), 6))
        seen_points.setdefault(key, []).append(i)
    spread = max(float(np.abs(coords).max()), 1.0)
    for idxs in seen_points.values():
        if len(idxs) < 2:
            continue
        for k, i in enumerate(idxs):
            h = int(hashlib.sha256(top_author_names[i].encode()).hexdigest(), 16)
            angle = (h % 360) * math.pi / 180
            radius = 0.02 * (k + 1) * spread
            coords[i, 0] += radius * math.cos(angle)
            coords[i, 1] += radius * math.sin(angle)

    grid = [[[float(cosine[i, j]), int(shared_counts[i, j])] for j in leaf_order_idx] for i in leaf_order_idx]
    tree_json = _linkage_to_tree(tree, [{"author": name} for name in top_author_names])

    from psycopg.types.json import Json

    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {s}.{cluster_table}")
        for i, author in enumerate(top_author_names):
            cur.execute(
                f"""INSERT INTO {s}.{cluster_table} (author, cluster_id, mds_x, mds_y, book_count, computed_at)
                    VALUES (%s,%s,%s,%s,%s, now())""",
                (author, int(cluster_ids[i]), float(coords[i, 0]), float(coords[i, 1]),
                 book_count_by_author[author]),
            )
        cur.execute(
            f"""INSERT INTO {s}.{cluster_run_table} (id, leaf_order, grid, tree_json, computed_at)
                VALUES (1, %s, %s, %s, now())
                ON CONFLICT (id) DO UPDATE SET
                    leaf_order=EXCLUDED.leaf_order, grid=EXCLUDED.grid,
                    tree_json=EXCLUDED.tree_json, computed_at=now()""",
            (leaf_order, Json(grid), Json(tree_json)),
        )
    conn.commit()
    return {"authors": n, "clusters": int(cluster_ids.max()) if n else 0}


def compute_book_clustering(conn, schema: str = DEFAULT_SCHEMA, *, top_n: int = 200,
                            max_df_fraction: float = 0.5, n_clusters: int = 12,
                            min_fame: float | None = None) -> dict:
    """`concordance book-clustering` / `maintain`'s clustering step, one
    level down from compute_author_clustering: the data behind the cluster
    map, similarity matrix, and dendrogram views for the top `top_n` books
    by (extracted-vocabulary) word count.

    Reuses the exact corpus-wide book-df IDF setup compute_book_similarity
    uses (same n_books/df/max_df_fraction computation over ALL books, not
    just the top_n) -- so "the why" (book_similarity's scores) and "the
    map" (cluster positions here) share one consistent notion of
    similarity; only WHICH books enter the pairwise/clustering step is
    restricted to top_n, not how a shared word is weighted. Unlike
    compute_author_clustering, there's no PLACEHOLDER_AUTHORS filter here --
    that's an author-level aggregation-label concept ("Various" isn't a
    real writer), and doesn't disqualify an individual book from having a
    real, clusterable vocabulary of its own; a book by a placeholder author
    just carries that through to book_cluster.author as-is (nullable,
    same as book.author itself).

    See compute_author_clustering's own docstring for the distance
    definition (sqrt(2*(1-cosine)), a proper Euclidean distance for
    L2-normalized vectors), classical MDS technique (direct eigh, no
    sklearn, deterministic axis-sign pinning), and jitter reasoning
    (identical qualifying word sets landing on the same point) -- all
    reused verbatim here, just keyed by book_id instead of author name.

    min_fame: see compute_author_clustering's own min_fame docstring --
    identical selection-and-destination-table swap (book_fame.fame_score
    >= min_fame instead of top_n by word count; writes book_cluster_fame/
    book_cluster_fame_run instead of book_cluster/book_cluster_run; not
    wired into `maintain`), one level down.

    Writes book_cluster and the singleton book_cluster_run in one
    transaction at the end, same all-or-nothing reasoning as
    compute_author_clustering (top_n=200 is small enough that one clean
    transaction is trivial, and the map/matrix/dendrogram must never
    disagree with each other)."""
    import hashlib
    import math

    import numpy as np
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform
    from scipy.sparse import csr_matrix

    s = _safe_schema(schema)
    cluster_table = "book_cluster_fame" if min_fame is not None else "book_cluster"
    cluster_run_table = "book_cluster_fame_run" if min_fame is not None else "book_cluster_run"

    with conn.cursor() as cur:
        cur.execute(f"""SELECT count(DISTINCT wb.book_id) FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id WHERE w.active""")
        n_books = cur.fetchone()[0]
        if n_books < 3:
            conn.commit()  # see compute_book_similarity's own early-return commit note
            return {"books": 0, "clusters": 0}

        cur.execute(f"""SELECT wb.word_id, count(DISTINCT wb.book_id) AS df
                        FROM {s}.word_book wb JOIN {s}.word w ON w.id = wb.word_id
                        WHERE w.active GROUP BY wb.word_id""")
        max_df = max_df_fraction * n_books
        idf = {wid: math.log(n_books / df) for wid, df in cur.fetchall() if df <= max_df}
        if not idf:
            conn.commit()
            return {"books": 0, "clusters": 0}

        if min_fame is not None:
            cur.execute(f"""SELECT b.id, b.title, b.author, count(DISTINCT w.id) AS word_count
                            FROM {s}.book b
                            JOIN {s}.word_book wb ON wb.book_id = b.id
                            JOIN {s}.word w ON w.id = wb.word_id AND w.active
                            JOIN {s}.book_fame bf ON bf.book_id = b.id
                            WHERE bf.fame_score >= %s
                            GROUP BY b.id, b.title, b.author, bf.fame_score
                            ORDER BY bf.fame_score DESC, word_count DESC""", (min_fame,))
        else:
            cur.execute(f"""SELECT b.id, b.title, b.author, count(DISTINCT w.id) AS word_count
                            FROM {s}.book b
                            JOIN {s}.word_book wb ON wb.book_id = b.id
                            JOIN {s}.word w ON w.id = wb.word_id AND w.active
                            GROUP BY b.id, b.title, b.author
                            ORDER BY word_count DESC
                            LIMIT %s""", (top_n,))
        top_books = cur.fetchall()
        top_book_ids = [r[0] for r in top_books]
        title_by_id = {r[0]: r[1] for r in top_books}
        author_by_id = {r[0]: r[2] for r in top_books}
        word_count_by_id = {r[0]: r[3] for r in top_books}

        if len(top_book_ids) < 3:
            conn.commit()
            return {"books": 0, "clusters": 0}

        cur.execute(f"""SELECT DISTINCT wb.word_id, wb.book_id FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        WHERE w.active AND wb.book_id = ANY(%s)
                          AND wb.word_id = ANY(%s)""",
                    (top_book_ids, list(idf.keys())))
        rows = cur.fetchall()

    n = len(top_book_ids)
    book_index = {bid: i for i, bid in enumerate(top_book_ids)}
    word_index: dict[int, int] = {}
    row_idx, col_idx, weighted_data = [], [], []
    for wid, bid in rows:
        j = word_index.setdefault(wid, len(word_index))
        row_idx.append(book_index[bid])
        col_idx.append(j)
        weighted_data.append(idf[wid])

    matrix = csr_matrix((weighted_data, (row_idx, col_idx)), shape=(n, len(word_index)))
    binary = csr_matrix(([1] * len(row_idx), (row_idx, col_idx)), shape=(n, len(word_index)))
    shared_counts = (binary @ binary.T).toarray().astype(int)

    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1))).reshape(-1)
    norms[norms == 0] = 1.0  # a book with zero qualifying words (all excluded by max_df) -- avoid /0
    normalized = matrix.multiply(1 / norms[:, None]).tocsr()
    cosine = (normalized @ normalized.T).toarray()
    np.clip(cosine, -1.0, 1.0, out=cosine)  # guard float round-off past 1 on a self-pair

    dist = np.sqrt(np.clip(2 * (1 - cosine), 0, None))
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    tree = linkage(condensed, method="ward", optimal_ordering=True)
    leaf_order_idx = leaves_list(tree)
    leaf_order_ids = [top_book_ids[i] for i in leaf_order_idx]
    cluster_ids = fcluster(tree, t=n_clusters, criterion="maxclust")

    # Classical (Torgerson) MDS: double-center the squared-distance matrix,
    # take the top-2 eigenvectors of the resulting Gram matrix.
    d2 = dist ** 2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ d2 @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    top2 = np.argsort(eigvals)[::-1][:2]
    coords = eigvecs[:, top2] * np.sqrt(np.clip(eigvals[top2], 0, None))

    for axis in range(coords.shape[1]):
        col = coords[:, axis]
        if col[np.argmax(np.abs(col))] < 0:
            coords[:, axis] = -col

    seen_points: dict[tuple, list[int]] = {}
    for i in range(n):
        key = (round(float(coords[i, 0]), 6), round(float(coords[i, 1]), 6))
        seen_points.setdefault(key, []).append(i)
    spread = max(float(np.abs(coords).max()), 1.0)
    for idxs in seen_points.values():
        if len(idxs) < 2:
            continue
        for k, i in enumerate(idxs):
            h = int(hashlib.sha256(str(top_book_ids[i]).encode()).hexdigest(), 16)
            angle = (h % 360) * math.pi / 180
            radius = 0.02 * (k + 1) * spread
            coords[i, 0] += radius * math.cos(angle)
            coords[i, 1] += radius * math.sin(angle)

    grid = [[[float(cosine[i, j]), int(shared_counts[i, j])] for j in leaf_order_idx] for i in leaf_order_idx]
    leaf_order = [{"id": bid, "title": title_by_id[bid], "author": author_by_id[bid]} for bid in leaf_order_ids]
    tree_json = _linkage_to_tree(
        tree, [{"id": bid, "title": title_by_id[bid], "author": author_by_id[bid]} for bid in top_book_ids]
    )

    from psycopg.types.json import Json

    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {s}.{cluster_table}")
        for i, bid in enumerate(top_book_ids):
            cur.execute(
                f"""INSERT INTO {s}.{cluster_table} (book_id, title, author, cluster_id, mds_x, mds_y,
                                                    word_count, computed_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s, now())""",
                (bid, title_by_id[bid], author_by_id[bid], int(cluster_ids[i]),
                 float(coords[i, 0]), float(coords[i, 1]), word_count_by_id[bid]),
            )
        cur.execute(
            f"""INSERT INTO {s}.{cluster_run_table} (id, leaf_order, grid, tree_json, computed_at)
                VALUES (1, %s, %s, %s, now())
                ON CONFLICT (id) DO UPDATE SET
                    leaf_order=EXCLUDED.leaf_order, grid=EXCLUDED.grid,
                    tree_json=EXCLUDED.tree_json, computed_at=now()""",
            (Json(leaf_order), Json(grid), Json(tree_json)),
        )
    conn.commit()
    return {"books": n, "clusters": int(cluster_ids.max()) if n else 0}
