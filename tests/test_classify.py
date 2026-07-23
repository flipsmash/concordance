"""Classifier code-validation and parsing (no model, no DB)."""

from __future__ import annotations

import os

import pytest

from concordance import classify, db

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


def test_validate_drops_hallucinated_and_repairs_subcodes():
    # G3.1/K5.3 aren't real -> repair to nearest valid ancestor; junk dropped; case fixed
    assert classify._validate(["G3.1", "K5.3", "m4", "junk", "E4.1"]) == ["G3", "K5", "M4"]


def test_validate_caps_at_three_and_dedupes():
    assert classify._validate(["A5.1", "A5.1", "B1", "C1", "E1"]) == ["A5.1", "B1", "C1"]


def test_validate_strips_usas_polarity_markers():
    # USAS marks polarity with +/- ; we ignore it for the code identity
    assert classify._validate(["A5.1+", "E4.1-"]) == ["A5.1", "E4.1"]


def test_validate_ignores_non_list():
    assert classify._validate("A1") == []
    assert classify._validate(None) == []


def test_parse_bare_array():
    assert classify._parse('[{"w":"cannon","c":["G3"]}]') == [{"w": "cannon", "c": ["G3"]}]


def test_parse_strips_fence_and_trailing_prose():
    raw = '```json\n[{"w":"x","c":["A1"]}]\n```\ndone'
    assert classify._parse(raw) == [{"w": "x", "c": ["A1"]}]


def test_parse_garbage_returns_empty():
    assert classify._parse("I cannot comply") == []


def test_prompt_items_injects_wnd_hint(monkeypatch):
    from concordance import wndomains
    monkeypatch.setattr(wndomains, "_lexicon", {"frigate": {"military", "nautical"}})
    items = classify._prompt_items([{"word": "frigate", "definition": "a warship", "sentence": "the frigate sailed"}])
    assert set(items[0]["hint"]) == {"G3", "M4"}     # WND prior surfaced as a hint


class _FakeClassifier:
    """Deterministic stand-in for Classifier -- no LLM, no GPU. Critical for
    these tests specifically: a real Classifier() would try to load the
    14B model, competing for the same GPU/VRAM a real `concordance
    classify`/`maintain` run might be using at the same time, exactly the
    resource-contention failure mode this project has hit live before.
    Tags every word with a fixed code so the interesting behavior under
    test is classify_and_store's own chunk/commit/resume plumbing, not
    the LLM's actual output."""
    def __init__(self, cfg=None):
        self.batch = 15
        self.seen_chunks: list[list[str]] = []

    def classify(self, items):
        self.seen_chunks.append([it["word"] for it in items])
        return {it["word"].lower(): ["A1"] for it in items}


def _seed_classify_schema(conn, schema, n_words):
    from concordance.model import Candidate

    db.apply_schema(conn, schema)
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {schema}.category (code, name, taxonomy, assignable) "
                    "VALUES ('A1', 'General and abstract terms', 'usas', true)")
    conn.commit()

    def word(lemma):
        c = Candidate(lemma=lemma, pos="NOUN")
        c.definition = f"definition of {lemma}"
        return c

    db.sync_book_results(conn, "Book", kept=[word(f"chunkword{i}") for i in range(n_words)],
                         rejected=[], schema=schema, author="Author")


@pg
def test_classify_and_store_commits_in_chunks_not_once_at_the_end(monkeypatch):
    schema = "cc_test_classify_chunking"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    _seed_classify_schema(conn, schema, 25)

    fake = _FakeClassifier()
    monkeypatch.setattr(classify, "Classifier", lambda cfg=None: fake)

    stats = classify.classify_and_store(conn, schema, only_missing=True, commit_every=10)
    assert stats["words"] == 25
    assert stats["classified"] == 25

    # 25 words at commit_every=10 -> 3 chunks (10, 10, 5), each its own
    # call to Classifier.classify -- not one call over all 25 the way the
    # old single-commit implementation made.
    assert [len(c) for c in fake.seen_chunks] == [10, 10, 5]

    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {schema}.word_category")
        assert cur.fetchone()[0] == 25

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_classify_and_store_only_missing_resumes_after_a_partial_run(monkeypatch):
    # Simulates exactly the crash-recovery scenario the periodic-commit
    # change exists for: a prior run committed some chunks then died
    # (crashed, killed, whatever) before finishing. A restart with
    # only_missing=True must pick up only the words still lacking a
    # word_category row -- not touch, and not re-send-to-the-classifier,
    # the ones a previous partial run already committed.
    schema = "cc_test_classify_resume"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    _seed_classify_schema(conn, schema, 10)

    with conn.cursor() as cur:
        cur.execute(f"select id, lemma from {schema}.word order by lemma")
        all_words = cur.fetchall()
        already_done = all_words[:4]  # simulate a prior run that committed these 4
        cur.execute(f"select id from {schema}.category where code='A1'")
        cat_id = cur.fetchone()[0]
        for word_id, _lemma in already_done:
            cur.execute(
                f"""INSERT INTO {schema}.word_category (word_id, category_id, confidence, source, is_primary)
                    VALUES (%s,%s,0.9,'llm',true)""",
                (word_id, cat_id),
            )
    conn.commit()

    fake = _FakeClassifier()
    monkeypatch.setattr(classify, "Classifier", lambda cfg=None: fake)

    stats = classify.classify_and_store(conn, schema, only_missing=True, commit_every=10)
    assert stats["words"] == 6   # 10 total - 4 already done
    assert stats["classified"] == 6

    seen_words = {w for chunk in fake.seen_chunks for w in chunk}
    already_done_lemmas = {lemma for _id, lemma in already_done}
    assert not (seen_words & already_done_lemmas)  # never re-sent to the "classifier"

    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {schema}.word_category")
        assert cur.fetchone()[0] == 10  # 4 pre-seeded + 6 newly classified

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()
