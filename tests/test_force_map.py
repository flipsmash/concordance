"""The force-directed maps (concordance/force_map.py, /api/browse/force-map)."""

from __future__ import annotations

import os
import shutil

import pytest

from concordance import db, force_map
from webapp.backend import auth

_URL = os.environ.get("CONCORDANCE_TEST_DB_URL", "")


def _connectable(url):
    try:
        import psycopg
        psycopg.connect(url, connect_timeout=3).close()
        return True
    except Exception:
        return False


pg = pytest.mark.skipif(not (_URL and _connectable(_URL)),
                        reason="set CONCORDANCE_TEST_DB_URL to a disposable Postgres to run")


def test_top_k_keeps_each_nodes_strongest_links_once():
    pairs = [("a", "b", 0.9), ("b", "a", 0.5), ("a", "c", 0.1), ("a", "d", 0.8), ("c", "d", 0.7)]
    edges = force_map._top_k(pairs, k=2)
    # a keeps b and d; c keeps d (and its only other link, a); d keeps a and c
    assert edges == {("a", "b"): 0.9, ("a", "d"): 0.8, ("c", "d"): 0.7, ("a", "c"): 0.1}


@pg
def test_author_force_map_end_to_end(monkeypatch):
    try:
        force_map.node_binary()
    except RuntimeError:
        pytest.skip("Node isn't installed")
    from starlette.testclient import TestClient

    from webapp.backend import main

    s = "cc_test_force_map"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
    conn.commit()
    db.apply_schema(conn, s)
    # two tight groups of authors, joined by one weak link
    groups = [[f"North, {i}" for i in range(6)], [f"South, {i}" for i in range(6)]]
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {s}.users (username, password_hash, is_admin) VALUES ('viewer', %s, false)",
                    (auth.hash_password("password123"),))
        for g in groups:
            for i, a in enumerate(g):
                for b in g[i + 1:]:
                    cur.execute(f"""INSERT INTO {s}.author_similarity (author_a, author_b, score, shared_word_count)
                                    VALUES (%s, %s, 0.9, 10)""", (a, b))
        cur.execute(f"""INSERT INTO {s}.author_similarity (author_a, author_b, score, shared_word_count)
                        VALUES ('North, 0', 'South, 0', 0.05, 1)""")
    conn.commit()

    monkeypatch.setattr(force_map, "_PASSES", [(200, dict(force_map._PASSES[0][1]))])
    stats = force_map.compute(conn, s, "author", progress=lambda m: None)
    assert stats["nodes"] == 12 and stats["regions"] >= 2

    old, main.SCHEMA = main.SCHEMA, s
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "viewer", "password": "password123"})
        body = client.get("/api/browse/force-map?entity=author").json()
        assert len(body["nodes"]["key"]) == 12 and len(body["edges"]["a"]) > 0
        regions = body["nodes"]["region"]
        north = {regions[i] for i, k in enumerate(body["nodes"]["key"]) if k.startswith("North")}
        south = {regions[i] for i, k in enumerate(body["nodes"]["key"]) if k.startswith("South")}
        assert len(north) == 1 and len(south) == 1 and north != south   # the two groups found
        assert client.get("/api/browse/force-map?entity=word").status_code == 404   # not laid out yet
        tip = client.get("/api/browse/force-map/detail?entity=author&key=North%2C%200").json()
        assert tip["title"] == "North, 0"
    finally:
        main.SCHEMA = old
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA {s} CASCADE")
        conn.commit()
        conn.close()


@pg
def test_word_colour_data_one_value_per_word():
    from webapp.backend.browse.force_map import _word_colour_data

    s = "cc_test_force_map_colour"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
    conn.commit()
    db.apply_schema(conn, s)
    with conn.cursor() as cur:
        ids = []
        for lemma, pos in (("susurrus", "noun"), ("sibilate", "verb"), ("tacitly", "adverb")):
            cur.execute(f"INSERT INTO {s}.word (lemma, definition, part_of_speech, active) VALUES (%s, 'x', %s, true) RETURNING id",
                        (lemma, pos))
            ids.append(cur.fetchone()[0])
        cur.execute(f"""INSERT INTO {s}.word_difficulty (word_id, difficulty, difficulty_factors)
                        VALUES (%s, 45.4, '{{"zipf": 1.25}}')""", (ids[0],))
        cur.execute(f"""INSERT INTO {s}.word_origin (word_id, root_lang, root_lang_name, family, route, method)
                        VALUES (%s, 'la', 'Latin', 'Latin', 'direct', 'wiktionary')""", (ids[0],))
        cur.execute(f"""INSERT INTO {s}.word_ngram (word_id, peak, recent, recency_ratio, peak_year, fetched_at)
                        VALUES (%s, 1.0, 0.5, 0.5, 1863, now()), (%s, 0, 0, NULL, NULL, now())""", (ids[0], ids[1]))
        out = _word_colour_data(cur, s, ids)
    assert out["difficulty"] == [45, None, None]
    assert out["frequency"] == [1.25, None, None]
    assert out["peak_year"] == [1863, None, None]          # absent from print -> unknown, not 0
    assert out["pos"] == ["n", "v", "r"]
    assert out["origins"] == ["Latin"] and out["origin"] == [0, None, None]
    assert len(out["domain"]) == 3 and len(out["domains"]) == 6
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {s} CASCADE")
    conn.commit()
    conn.close()
