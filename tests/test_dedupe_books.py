"""Removing duplicate books reversibly (concordance/dedupe_books.py).
DB-backed; needs CONCORDANCE_TEST_DB_URL like test_definition_edit.py."""

from __future__ import annotations

import os

import pytest

from concordance import db, dedupe_books as dd

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


def _state(cur, s):
    out = {}
    for q in (f"SELECT id, title, author, word_count FROM {s}.book ORDER BY id",
              f"SELECT word_id, book_id FROM {s}.word_book ORDER BY 1, 2",
              f"SELECT book_id, lemma, lemma_lc, reason FROM {s}.rejected_word ORDER BY 1, 2",
              f"SELECT book_id, genre FROM {s}.book_genre ORDER BY 1, 2",
              f"SELECT book_a_id, book_b_id FROM {s}.book_similarity ORDER BY 1, 2"):
        cur.execute(q)
        out[q.split("FROM ")[1].split()[0]] = cur.fetchall()
    return out


@pg
def test_finds_conservative_duplicates_and_removes_one_reversibly():
    s = "cc_test_dedupe_books"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
    conn.commit()
    db.apply_schema(conn, s)
    with conn.cursor() as cur:
        books = {}
        for title, author, wc in [
            ("The Divine Comedy, Paradise (Complete)", "Dante Alighieri", 38796),
            ("The Divine Comedy, Paradise, Complete", "Dante Alighieri", 38623),
            ("Coniston (Complete)", "Churchill, Winston", 64772),     # sizes too far apart
            ("Coniston — Complete", "Churchill, Winston", 174637),
            ("Poems", "Various", 1000), ("Poems.", "Various", 1001),  # placeholder author
        ]:
            cur.execute(f"INSERT INTO {s}.book (title, author, word_count) VALUES (%s, %s, %s) RETURNING id",
                        (title, author, wc))
            books[title + author] = cur.fetchone()[0]
        a, b = books["The Divine Comedy, Paradise (Complete)Dante Alighieri"], books["The Divine Comedy, Paradise, CompleteDante Alighieri"]
        words = []
        for lemma in ("empyrean", "seraph", "lucent"):
            cur.execute(f"INSERT INTO {s}.word (lemma, definition, active) VALUES (%s, 'x', true) RETURNING id", (lemma,))
            words.append(cur.fetchone()[0])
        # a has two words, b has one of those plus one of its own
        cur.executemany(f"INSERT INTO {s}.word_book (word_id, book_id) VALUES (%s, %s)",
                        [(words[0], a), (words[1], a), (words[1], b), (words[2], b)])
        cur.execute(f"INSERT INTO {s}.rejected_word (book_id, lemma, reason) VALUES (%s, 'Beatrice', 'proper_noun')", (b,))
        cur.execute(f"INSERT INTO {s}.book_genre (book_id, genre, source) VALUES (%s, 'Poetry', 'llm')", (b,))
        cur.execute(f"INSERT INTO {s}.book_similarity (book_a_id, book_b_id, score, shared_word_count) VALUES (%s, %s, 0.9, 2)", (a, b))
    conn.commit()

    groups = dd.find_duplicates(conn, s)
    assert [(k.id, [d.id for d in ds]) for k, ds in groups] == [(a, [b])]   # a has more words
    with conn.cursor() as cur:
        before = _state(cur, s)

    rid = dd.retire(conn, s, groups[0][0], groups[0][1][0], by="test", reason="same work")
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {s}.book WHERE id = %s", (b,))
        assert cur.fetchone() == (0,)
        cur.execute(f"SELECT book_id FROM {s}.word_book WHERE word_id = %s", (words[2],))
        assert cur.fetchall() == [(a,)]                  # b's own word kept a book
        cur.execute(f"SELECT status, added_links FROM {s}.book_retirement WHERE id = %s", (rid,))
        assert cur.fetchone() == ("removed", [words[2]])

    dd.restore(conn, s, rid)
    conn.commit()
    with conn.cursor() as cur:
        assert _state(cur, s) == before                  # exactly as it was
        cur.execute(f"SELECT status FROM {s}.book_retirement WHERE id = %s", (rid,))
        assert cur.fetchone() == ("restored",)
    with pytest.raises(ValueError):
        dd.restore(conn, s, rid)

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {s} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_contained_pass_removes_only_volumes_the_complete_book_holds(tmp_path):
    s = "cc_test_dedupe_contained"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
    conn.commit()
    db.apply_schema(conn, s)
    vol1 = "chapter one: the lighthouse keeper rowed toward the darkening headland " * 40
    vol2 = "chapter two: brambles overgrew the ruined chapel beyond the marshes " * 40
    vol3 = "chapter three: smugglers unloaded brandy beneath the crumbling viaduct " * 40
    texts = {
        "Grania — Complete": vol1 + vol2,           # Gutenberg's one-file edition, lacks volume 3
        "Grania — Volume 01": vol1,
        "Grania — Volume 02": vol2,
        "Grania — Volume 03": vol3,                # not in the Complete book: stays
        "Grania (Complete)": vol2,                 # our partial compilation: inside the fuller one
    }
    ids = {}
    with conn.cursor() as cur:
        for title, text in texts.items():
            p = tmp_path / f"{len(ids)}.txt"
            p.write_text(text)
            cur.execute(f"INSERT INTO {s}.book (title, author, word_count, archive_path) VALUES (%s, 'Lawless, Emily', %s, %s) RETURNING id",
                        (title, len(text.split()), str(p)))
            ids[title] = cur.fetchone()[0]
    conn.commit()

    pairs = dd.find_contained(conn, s)
    removed = sorted(d.title for _, d, _ in pairs)
    assert removed == ["Grania (Complete)", "Grania — Volume 01", "Grania — Volume 02"]
    assert {k.title for k, _, _ in pairs} == {"Grania — Complete"}

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {s} CASCADE")
    conn.commit()
    conn.close()
