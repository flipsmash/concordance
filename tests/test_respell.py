"""An admin's respelling of a word (concordance/respell.py and
POST /api/words/{id}/spelling). DB-backed tests need CONCORDANCE_TEST_DB_URL
(a disposable Postgres), as in test_definition_edit.py."""

from __future__ import annotations

import os

import pytest

from concordance import db, respell
from concordance.model import Candidate, Occurrence
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


def _schema(name: str):
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE")
    conn.commit()
    db.apply_schema(conn, name)
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {name}.users (username, password_hash, is_admin) VALUES ('adminuser', %s, true)",
                    (auth.hash_password("password123"),))
        cur.execute(f"INSERT INTO {name}.book (title) VALUES ('Book One'), ('Book Two') RETURNING id")
        books = [r[0] for r in cur.fetchall()]
    conn.commit()
    return conn, books


def _word(cur, s, lemma, definition="", **cols):
    names = ["lemma", "definition", "active"] + list(cols)
    cur.execute(f"INSERT INTO {s}.word ({', '.join(names)}) VALUES ({', '.join(['%s'] * len(names))}) RETURNING id",
                [lemma, definition, True, *cols.values()])
    return cur.fetchone()[0]


def _fake_lookup(definitions):
    def enrich(cand, **_):
        cand.definition = definitions.get(cand.lemma, "")
        cand.part_of_speech = "noun"
        cand.definition_source = "Local Wiktionary (DB)"
        return False
    return enrich


def test_apply_aliases_maps_an_old_spelling_onto_its_word():
    def cand(lemma, n):
        return Candidate(lemma=lemma, pos="NOUN",
                         occurrences=[Occurrence(sentence=f"{lemma} {i}", chapter="", surface=lemma) for i in range(n)])

    alone = {"zirbájeh": cand("zirbájeh", 2), "stew": cand("stew", 1)}
    assert respell.apply_aliases(alone, {"zirbájeh": "zirbaja"}) == 1
    assert sorted(alone) == ["stew", "zirbaja"] and alone["zirbaja"].lemma == "zirbaja"

    both = {"zirbájeh": cand("zirbájeh", 2), "zirbaja": cand("zirbaja", 1)}
    respell.apply_aliases(both, {"zirbájeh": "zirbaja"})
    assert list(both) == ["zirbaja"] and both["zirbaja"].count == 3


@pg
def test_respell_renames_looks_up_again_and_queues_the_upkeep(monkeypatch):
    from concordance import pipeline

    s = "cc_test_respell_rename"
    conn, books = _schema(s)
    with conn.cursor() as cur:
        wid = _word(cur, s, "zirbájeh", flagged_undefined=True, validity_label="likely-artifact",
                    validity_notes="absent from Google Books", ipa="zɪr")
        cur.execute(f"INSERT INTO {s}.word_book (word_id, book_id) VALUES (%s, %s)", (wid, books[0]))
        cur.execute(f"INSERT INTO {s}.word_origin VALUES (%s, 'fa', 'Persian', 'Arabic, Persian & Turkic', "
                    f"'direct', 'wiktionary')", (wid,))
    conn.commit()
    monkeypatch.setattr(pipeline, "_enrich_one", _fake_lookup({"zirbaja": "A Persian stew."}))

    out = respell.respell(conn, s, wid, "zirbaja", by="adminuser")
    assert out == {"word_id": wid, "lemma": "zirbaja", "merged": False, "definition_refreshed": True}
    with conn.cursor() as cur:
        cur.execute(f"""SELECT lemma, definition, flagged_undefined, validity_label, ipa, admin_suggested
                        FROM {s}.word WHERE id = %s""", (wid,))
        assert cur.fetchone() == ("zirbaja", "A Persian stew.", False, None, None, True)
        cur.execute(f"SELECT count(*) FROM {s}.word_origin WHERE word_id = %s", (wid,))
        assert cur.fetchone() == (0,)                       # cleared for the job to redo
        cur.execute(f"SELECT count(*) FROM {s}.word_book WHERE word_id = %s", (wid,))
        assert cur.fetchone() == (1,)                       # its book stays
        cur.execute(f"SELECT alias_lc, word_id FROM {s}.word_alias")
        assert cur.fetchall() == [("zirbájeh", wid)]
        cur.execute(f"SELECT word_id, old_lemma, new_lemma, status FROM {s}.word_refresh_job")
        assert cur.fetchall() == [(wid, "zirbájeh", "zirbaja", "pending")]

    # the old spelling names it now; respelling back drops that alias
    with pytest.raises(respell.RespellCollision):
        with conn.cursor() as cur:
            other = _word(cur, s, "kisel", "A dessert.")
        conn.commit()
        respell.respell(conn, s, other, "zirbájeh", by="adminuser")
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {s} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_respell_keeps_a_real_definition():
    s = "cc_test_respell_keep_def"
    conn, _ = _schema(s)
    with conn.cursor() as cur:
        wid = _word(cur, s, "colour", "A visual attribute.")
    conn.commit()
    out = respell.respell(conn, s, wid, "color", by="adminuser")
    assert out["definition_refreshed"] is False
    with conn.cursor() as cur:
        cur.execute(f"SELECT definition FROM {s}.word WHERE id = %s", (wid,))
        assert cur.fetchone() == ("A visual attribute.",)
        cur.execute(f"DROP SCHEMA {s} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_respell_into_an_existing_word_needs_merge_then_moves_everything():
    s = "cc_test_respell_merge"
    conn, books = _schema(s)
    with conn.cursor() as cur:
        src = _word(cur, s, "zirbájeh")
        dst = _word(cur, s, "zirbaja", "A Persian stew.")
        cur.execute(f"INSERT INTO {s}.word_book (word_id, book_id) VALUES (%s, %s), (%s, %s), (%s, %s)",
                    (src, books[0], src, books[1], dst, books[1]))
        cur.execute(f"SELECT id FROM {s}.users")
        uid = cur.fetchone()[0]
        cur.execute(f"INSERT INTO {s}.word_set (user_id, name) VALUES (%s, 'stews') RETURNING id", (uid,))
        set_id = cur.fetchone()[0]
        cur.execute(f"INSERT INTO {s}.word_set_item (set_id, word_id) VALUES (%s, %s)", (set_id, src))
        cur.execute(f"""INSERT INTO {s}.word_review_schedule (user_id, word_id, correct_count, incorrect_count)
                        VALUES (%s, %s, 3, 1), (%s, %s, 1, 0)""", (uid, src, uid, dst))
    conn.commit()

    with pytest.raises(respell.RespellCollision) as exc:
        respell.respell(conn, s, src, "Zirbaja", by="adminuser")
    assert exc.value.collision.word_id == dst and exc.value.collision.books == 1

    out = respell.respell(conn, s, src, "Zirbaja", by="adminuser", merge=True)
    assert out == {"word_id": dst, "lemma": "zirbaja", "merged": True, "definition_refreshed": False}
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {s}.word WHERE id = %s", (src,))
        assert cur.fetchone() == (0,)
        cur.execute(f"SELECT book_id FROM {s}.word_book WHERE word_id = %s ORDER BY book_id", (dst,))
        assert [r[0] for r in cur.fetchall()] == books
        cur.execute(f"SELECT word_id FROM {s}.word_set_item WHERE set_id = %s", (set_id,))
        assert cur.fetchall() == [(dst,)]
        cur.execute(f"SELECT correct_count, incorrect_count FROM {s}.word_review_schedule WHERE word_id = %s", (dst,))
        assert cur.fetchone() == (3, 1)                     # the busier schedule wins
        cur.execute(f"SELECT alias_lc, word_id FROM {s}.word_alias")
        assert cur.fetchall() == [("zirbájeh", dst)]
        cur.execute(f"DROP SCHEMA {s} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_spelling_api_reports_a_collision_then_merges(monkeypatch):
    from starlette.testclient import TestClient

    from webapp.backend import main, respell_api

    s = "cc_test_respell_api"
    conn, _ = _schema(s)
    with conn.cursor() as cur:
        src = _word(cur, s, "zirbájeh")
        dst = _word(cur, s, "zirbaja", "A Persian stew.")
        cur.execute(f"UPDATE {s}.word SET active = false WHERE id = %s", (dst,))
    conn.commit()
    started = []
    monkeypatch.setattr(respell_api, "start_refresh_job", lambda: started.append(1))
    old_schema, main.SCHEMA = main.SCHEMA, s
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.post(f"/api/words/{src}/spelling", json={"spelling": "zirbaja"})
        assert res.status_code == 409
        assert res.json()["detail"]["collision"] == {
            "word_id": dst, "lemma": "zirbaja", "active": False, "inactive_reason": "pruned by hand", "books": 0}
        assert client.post(f"/api/words/{src}/spelling", json={"spelling": "zir baja 2"}).status_code == 400

        res = client.post(f"/api/words/{src}/spelling", json={"spelling": "zirbaja", "merge": True})
        assert res.status_code == 200 and res.json()["word_id"] == dst
        assert started == [1]
        detail = client.get(f"/api/words/{dst}").json()        # brought back by the merge
        assert detail["aliases"] == ["zirbájeh"] and detail["refreshing"] is True
    finally:
        main.SCHEMA = old_schema
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA {s} CASCADE")
        conn.commit()
        conn.close()
