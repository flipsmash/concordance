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
