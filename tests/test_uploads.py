"""Admin book uploads (webapp/backend/uploads_api.py) and the background job
that imports them (concordance/uploads.py). DB-backed; needs
CONCORDANCE_TEST_DB_URL like test_definition_edit.py."""

from __future__ import annotations

import os

import pytest

from concordance import db, uploads
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


def _schema(name):
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE")
    conn.commit()
    db.apply_schema(conn, name)
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {name}.users (username, password_hash, is_admin) VALUES ('adminuser', %s, true)",
                    (auth.hash_password("password123"),))
    conn.commit()
    return conn


@pg
def test_upload_api_stores_queues_and_guards(monkeypatch, tmp_path):
    from starlette.testclient import TestClient

    from webapp.backend import main, uploads_api

    s = "cc_test_uploads_api"
    conn = _schema(s)
    started = []
    monkeypatch.setattr(uploads_api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(uploads_api.jobs, "start", lambda cmd, schema: started.append(cmd) or True)
    old, main.SCHEMA = main.SCHEMA, s
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        def post(name, data, title="Frankenstein: or, The Modern Prometheus", **extra):
            return client.post("/api/admin/uploads", files={"file": (name, data)},
                               data={"title": title, "author": "Shelley, Mary", **extra})

        res = post("frankenstein.txt", b"It was on a dreary night of November.")
        assert res.status_code == 200, res.text
        up = res.json()
        assert (up["status"], up["title"], up["author"]) == ("queued", "Frankenstein: or, The Modern Prometheus",
                                                             "Shelley, Mary")
        assert started == ["process-uploads"]
        stored = list(tmp_path.iterdir())
        assert len(stored) == 1 and stored[0].suffix == ".txt"     # a generated name, not the upload's

        assert post("notes.docx", b"x").status_code == 400
        assert post("fake.pdf", b"not a pdf", title="Other").status_code == 400
        assert post("empty.txt", b"", title="Empty").status_code == 400
        assert len(list(tmp_path.iterdir())) == 1                   # rejected files aren't kept

        # same title while the first still waits: refused outright
        assert post("again.txt", b"text", title="frankenstein: OR, the modern prometheus").status_code == 409
        with conn.cursor() as cur:
            cur.execute(f"UPDATE {s}.book_upload SET status = 'done'")
            cur.execute(f"INSERT INTO {s}.book (title, author) VALUES ('Frankenstein: or, The Modern Prometheus', 'Shelley, Mary')")
        conn.commit()
        res = post("again.txt", b"text")
        assert res.status_code == 409 and res.json()["detail"]["existing"]["title"].startswith("Frankenstein")
        assert post("again.txt", b"text", replace="true").status_code == 200

        listed = client.get("/api/admin/uploads").json()
        assert [u["status"] for u in listed] == ["queued", "done"]
    finally:
        main.SCHEMA = old
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA {s} CASCADE")
        conn.commit()
        conn.close()


@pg
def test_process_uploads_ingests_then_runs_the_upkeep_once(monkeypatch, tmp_path):
    s = "cc_test_uploads_job"
    conn = _schema(s)
    with conn.cursor() as cur:
        for title in ("Good Book", "Scanned Book"):
            cur.execute(f"""INSERT INTO {s}.book_upload (filename, stored_path, title, author)
                            VALUES (%s, %s, %s, 'Someone, A')""", (f"{title}.txt", str(tmp_path / title), title))
    conn.commit()

    ran = []

    def fake_run(step, args, schema, log_dir, log=print, database_url=None):
        ran.append(step)
        if args[0] == "ingest":
            if "Scanned Book" in args:
                return False, "this PDF has no text layer (scanned?)"
            with conn.cursor() as cur:
                cur.execute(f"INSERT INTO {schema}.book (title, author) VALUES (%s, 'Someone, A') RETURNING id",
                            (args[args.index("--title") + 1],))
                cur.execute(f"INSERT INTO {schema}.word (lemma, definition, active) VALUES ('zirbaja', 'A stew.', true) "
                            f"RETURNING id")
                wid = cur.fetchone()[0]
                cur.execute(f"INSERT INTO {schema}.word_book (word_id, book_id) SELECT %s, id FROM {schema}.book", (wid,))
            conn.commit()
            return True, ""
        return step != "book-fame", "fame lookup timed out"     # one upkeep step fails, non-fatally

    monkeypatch.setattr(uploads, "_run", fake_run)
    out = uploads.process_uploads(conn, s, database_url=_URL, archive_dir=tmp_path, log_dir=tmp_path,
                                  log=lambda m: None)
    assert out == {"uploads": 1}
    assert ran[:2] == ["ingest_1", "ingest_2"]
    assert ran[2:] == [name for name, _ in uploads.UPKEEP + uploads.RELATEDNESS]   # once for the batch
    with conn.cursor() as cur:
        cur.execute(f"SELECT title, status, words_kept, error FROM {s}.book_upload ORDER BY id")
        good, scanned = cur.fetchall()
    assert good[:3] == ("Good Book", "done", 1) and "book-fame failed" in good[3]
    assert scanned[1] == "failed" and "no text layer" in scanned[3]

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {s} CASCADE")
    conn.commit()
    conn.close()
