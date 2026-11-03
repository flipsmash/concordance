import numpy as np

from concordance import corpus_map


def test_cluster_count_scales_with_n():
    assert corpus_map.n_clusters(29062) == 43
    assert corpus_map.n_clusters(800) == 7
    assert corpus_map.n_clusters(3) == 1


def test_build_matrix_drops_words_in_over_half_the_rows():
    pairs = [(1, 10), (2, 10), (3, 10), (1, 20), (2, 30), (3, 30), (3, 40)]
    keys, words, X, nwords, B = corpus_map.build_matrix(pairs)
    assert keys == [1, 2, 3] and list(words) == [10, 20, 30, 40]
    # with 3 rows, words 10 (all rows) and 30 (two rows) are each in over half
    assert X[:, 0].nnz == 0 and X[:, 2].nnz == 0
    assert list(nwords) == [1, 0, 1]                          # row 2 has nothing distinctive left
    assert np.allclose(np.sqrt(X.multiply(X).sum(1)).A.ravel()[[0, 2]], 1.0)


def test_label_words_need_a_minimum_share():
    from scipy.sparse import csr_matrix
    # 40 members: word 0 used by all of them, word 1 by a single member only
    B = csr_matrix(np.array([[1, 1 if i == 0 else 0, 0] for i in range(40)] + [[0, 0, 1]] * 60, dtype=np.float32))
    share_all = np.asarray(B.sum(0)).ravel() / 100
    words = corpus_map.label_words(B, np.arange(40), share_all, ["wigwam", "abacination", "other"])
    assert words == ["wigwam"]


import os
import pytest

from concordance import db

_URL = os.environ.get("CONCORDANCE_TEST_DB_URL")
pg = pytest.mark.skipif(not _URL, reason="no test database")


@pg
def test_compute_maps_every_book_and_author():
    schema = "cc_test_corpus_map"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    sea = [f"sea{i}" for i in range(30)]
    west = [f"west{i}" for i in range(30)]
    with conn.cursor() as cur:
        for w in sea + west + ["common"]:
            cur.execute(f"INSERT INTO {schema}.word (lemma, active) VALUES (%s, true)", (w,))
        rng = np.random.default_rng(0)
        for b in range(40):
            vocab, author = (sea, f"Mariner {b % 4}") if b < 20 else (west, f"Rider {b % 4}")
            cur.execute(f"INSERT INTO {schema}.book (title, author) VALUES (%s, %s) RETURNING id",
                        (f"Book {b}", author if b != 39 else "Various"))
            bid = cur.fetchone()[0]
            for w in list(rng.choice(vocab, 12, replace=False)) + ["common"]:
                cur.execute(f"""INSERT INTO {schema}.word_book (word_id, book_id)
                                SELECT id, %s FROM {schema}.word WHERE lemma = %s""", (bid, w))
    conn.commit()

    st = corpus_map.compute(conn, schema, "book", progress=lambda *_: None)
    assert st["n"] == 40 and st["clusters"] == corpus_map.n_clusters(40)
    sa = corpus_map.compute(conn, schema, "author", progress=lambda *_: None)
    assert sa["n"] == 8                                         # "Various" left out
    with conn.cursor() as cur:
        cur.execute(f"SELECT label, x, y, cluster, sparse FROM {schema}.corpus_map_point WHERE entity='book'")
        rows = cur.fetchall()
        assert all(0 <= x <= corpus_map.COORD_SCALE and 0 <= y <= corpus_map.COORD_SCALE for _, x, y, _, _ in rows)
        assert all(sp for *_, sp in rows)                       # 12 words each: under SPARSE_WORDS
        sea_c = {c for l, _, _, c, _ in rows if int(l.split()[1]) < 20}
        west_c = {c for l, _, _, c, _ in rows if int(l.split()[1]) >= 20}
        assert not sea_c & west_c                               # the two genres never share a cluster
        cur.execute(f"SELECT label_words FROM {schema}.corpus_map_cluster WHERE entity='book' AND subcluster=-1")
        for (words,) in cur.fetchall():
            assert words and "common" not in words              # in every book: never a label
            assert len({w[:3] for w in words}) == 1             # all from one genre
        cur.execute(f"SELECT entity, n FROM {schema}.corpus_map_run ORDER BY entity")
        assert cur.fetchall() == [("author", 8), ("book", 40)]
    # a re-run replaces the rows rather than adding to them
    corpus_map.compute(conn, schema, "book", progress=lambda *_: None)
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {schema}.corpus_map_point WHERE entity='book'")
        assert cur.fetchone()[0] == 40
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
