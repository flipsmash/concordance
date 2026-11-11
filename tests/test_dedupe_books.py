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


# --- third pass: by text --------------------------------------------------------

def _story(seed: int, n: int = 30000) -> str:
    import random
    rnd = random.Random(seed)
    vocab = ["".join(rnd.choice("abcdefghijklmnoprstuvwy") for _ in range(rnd.randint(3, 9))) for _ in range(4000)]
    return " ".join(rnd.choice(vocab) for _ in range(n))


def test_title_pieces():
    assert dd._piece_of("Petty Troubles of Married Life, Second Part", "Petty Troubles of Married Life, Complete")
    assert dd._piece_of("My Mark Twain (from Literary Friends and Acquaintance)",
                        "Literary Friends and Acquaintance; a Personal Retrospect of American Authorship")
    assert dd._piece_of("Thomas Wingfold, Curate V2", "Thomas Wingfold, Curate")
    assert not dd._piece_of("Being a Boy", "The Complete Writings of Charles Dudley Warner — Volume 4")
    assert not dd._piece_of("Ballads and Poems of Tragic Life", "Poems (Complete)")      # one shared word
    assert not dd._piece_of("The Physiology of Marriage, Complete", "Analytical Studies")  # "Complete" alone


def test_by_text_finds_retitled_copies_volumes_and_covered_compilations(tmp_path):
    a, b, c, v1, v2 = (_story(i) for i in range(5))
    texts = {
        1: ("A Christmas Carol", a), 2: ("A Christmas Carol in Prose; Being a Ghost Story", a),   # same text
        3: ("The Bells", b), 4: ("Stories: The Carol and The Bells", a + " " + b),               # covered compilation
        5: ("Entire Works", a + " " + b + " " + c),                                               # holds c: stays
        6: ("Queechy", v1 + " " + v2), 7: ("Queechy, Volume I", v1), 8: ("Queechy, Volume II", v2),
    }
    sk, books = {}, {}
    for bid, (title, text) in texts.items():
        p = tmp_path / f"{bid}.txt"
        p.write_text(text)
        sk[bid] = dd.sketch_text(str(p))
        books[bid] = dd.Book(bid, title, "Someone", len(text.split()), linked_words=10 - bid)
    overlaps = dd.text_overlaps(sk)
    same = dd.find_same_text(books, overlaps)
    assert [(k.id, [d.id for d in ds]) for k, ds in same] == [(1, [2])]          # more linked words stays
    gone = {2: 1}
    found = dd.find_containers({i: x for i, x in books.items() if i not in gone}, overlaps, sk, gone)
    splits = {(f[1].id, tuple(v.id for v in f[2])) for f in found if f[0] == "split"}
    comps = {f[1].id: {p.id for p in f[2]} for f in found if f[0] == "compilation"}
    assert splits == {(6, (7, 8))}
    assert 5 not in comps                       # c is in no other book: the Entire Works stay
    assert comps.get(4) == {1, 3}


@pg
def test_retiring_a_compilation_spreads_its_links_and_restores_exactly(tmp_path):
    s = "cc_test_dedupe_compilation"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {s} CASCADE")
    conn.commit()
    db.apply_schema(conn, s)
    texts = {"Tales of All Countries": "the mistletoe bough and the parson's daughter, with a rare quillon",
             "The Mistletoe Bough": "the mistletoe bough, a quillon gleaming",
             "The Parson's Daughter": "the parson's daughter of oxney colne"}
    ids = {}
    with conn.cursor() as cur:
        for title, text in texts.items():
            p = tmp_path / f"{len(ids)}.txt"
            p.write_text(text)
            cur.execute(f"INSERT INTO {s}.book (title, author, word_count, archive_path) VALUES (%s, 'Trollope, Anthony', %s, %s) RETURNING id",
                        (title, len(text.split()), str(p)))
            ids[title] = cur.fetchone()[0]
        comp, bough, daughter = ids["Tales of All Countries"], ids["The Mistletoe Bough"], ids["The Parson's Daughter"]
        words = {}
        for lemma in ("quillon", "colne", "mistletoe"):
            cur.execute(f"INSERT INTO {s}.word (lemma, definition, active) VALUES (%s, 'x', true) RETURNING id", (lemma,))
            words[lemma] = cur.fetchone()[0]
        # quillon and colne are linked only to the compilation; mistletoe to the story too
        cur.executemany(f"INSERT INTO {s}.word_book (word_id, book_id) VALUES (%s, %s)",
                        [(words["quillon"], comp), (words["colne"], comp), (words["mistletoe"], comp),
                         (words["mistletoe"], bough)])
    conn.commit()
    with conn.cursor() as cur:
        before = _state(cur, s)

    parts = [dd.book(conn, s, bough), dd.book(conn, s, daughter)]
    spread = dd.spread_links(conn, s, dd.book(conn, s, comp), parts)
    # each orphaned word goes to the story whose text has it; mistletoe already has a book
    assert spread == {bough: [words["quillon"]], daughter: [words["colne"]]}
    rid = dd.retire(conn, s, parts[0], dd.book(conn, s, comp), by="test", reason="compilation", spread=spread)
    conn.commit()
    with conn.cursor() as cur:
        cur.execute(f"SELECT word_id, book_id FROM {s}.word_book ORDER BY 1, 2")
        assert sorted(cur.fetchall()) == sorted([(words["quillon"], bough), (words["colne"], daughter),
                                                 (words["mistletoe"], bough)])
    dd.restore(conn, s, rid)
    conn.commit()
    with conn.cursor() as cur:
        assert _state(cur, s) == before
        cur.execute(f"DROP SCHEMA {s} CASCADE")
    conn.commit()
    conn.close()
