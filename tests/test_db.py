"""Postgres sync. Pure helpers run always; the round-trip test runs only when a
throwaway DB is provided via CONCORDANCE_TEST_DB_URL (else skipped)."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from concordance import db
from concordance.master import MASTER_COLUMNS


# --- pure helpers (no database) -------------------------------------------

def test_synonyms_and_books_split():
    assert db._synonyms("a; b ;c") == ["a", "b", "c"]
    assert db._synonyms("") == []
    assert db._books("BookA; BookB") == ["BookA", "BookB"]


def test_safe_schema_rejects_injection():
    assert db._safe_schema("concordance") == "concordance"
    for bad in ["public; drop table x", "a-b", "1abc", "a b", ""]:
        with pytest.raises(ValueError):
            db._safe_schema(bad)


def test_read_master_rows_keeps_master_columns(tmp_path):
    p = tmp_path / "master_vocab.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
        w.writeheader()
        row = {c: "" for c in MASTER_COLUMNS}
        row.update(word="cangue", date_added="2026-07-05", source_book="BookA; BookB")
        w.writerow(row)
    rows = db._read_master_rows(p)
    assert rows[0]["source_book"] == "BookA; BookB"   # NOT dropped
    assert rows[0]["date_added"] == "2026-07-05"


# --- round trip (needs a real, disposable Postgres) -----------------------

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


@pg
def test_ingest_never_clobbers_a_definition_and_flags_undefined(tmp_path):
    from concordance.model import Candidate

    schema = "cc_test2"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    # Book 1: "cangue" is successfully defined.
    defined = Candidate(lemma="cangue", pos="NOUN")
    defined.definition = "a wooden collar"
    defined.definition_source = "Local Wiktionary (DB)"
    db.sync_book_results(conn, "Book One", kept=[defined], rejected=[], schema=schema)

    with conn.cursor() as cur:
        cur.execute(f"select definition, flagged_undefined from {schema}.word where lemma='cangue'")
        row = cur.fetchone()
        assert row == ("a wooden collar", False)

    # Book 2: same word recurs but this time enrichment fails (blank definition)
    # — the existing definition must survive, not be clobbered to blank.
    undefined_repeat = Candidate(lemma="cangue", pos="NOUN")
    db.sync_book_results(conn, "Book Two", kept=[undefined_repeat], rejected=[], schema=schema)

    with conn.cursor() as cur:
        cur.execute(f"select definition, flagged_undefined from {schema}.word where lemma='cangue'")
        row = cur.fetchone()
        assert row == ("a wooden collar", False)   # not clobbered, not flagged (still defined)

    # A brand-new word that comes in with no definition at all must be flagged,
    # and stay flagged even if refill later fills it in (sticky by design).
    never_defined = Candidate(lemma="fuligin", pos="NOUN")
    db.sync_book_results(conn, "Book One", kept=[never_defined], rejected=[], schema=schema)

    with conn.cursor() as cur:
        cur.execute(f"select definition, flagged_undefined, flagged_undefined_at "
                    f"from {schema}.word where lemma='fuligin'")
        d, flagged, flagged_at = cur.fetchone()
        assert d == "" and flagged is True and flagged_at is not None

    with conn.cursor() as cur:
        cur.execute(f"update {schema}.word set definition='a fictional black pigment' "
                    f"where lemma='fuligin'")
        cur.execute(f"select definition, flagged_undefined from {schema}.word where lemma='fuligin'")
        assert cur.fetchone() == ("a fictional black pigment", True)   # flag persists

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_ingest_invalidates_stale_definition_dependents_on_change():
    """The "changeful" bug: a word's quiz_definition/categories/embedding get
    computed once, then the same lemma resolves to a DIFFERENT dictionary
    sense on a later book's ingest -- definition changes, but nothing used to
    tell the downstream only-missing-gated artifacts to recompute, so they
    silently kept describing the old text. sync_book_results/sync_master
    should now clear them whenever an upsert actually changes an existing
    definition (never on a first-time fill, never when it's unchanged)."""
    from pgvector.psycopg import register_vector

    from concordance.model import Candidate

    schema = "cc_test_definition_invalidation"
    conn = db.connect(_URL)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    first = Candidate(lemma="changeful", pos="ADJ")
    first.definition = "very susceptible to change; changing frequently"
    db.sync_book_results(conn, "Book One", kept=[first], rejected=[], schema=schema)

    with conn.cursor() as cur:
        cur.execute(f"SELECT id FROM {schema}.word WHERE lemma='changeful'")
        word_id = cur.fetchone()[0]
        # Simulate a maintenance pass having already run on the ORIGINAL definition.
        cur.execute(f"UPDATE {schema}.word SET quiz_definition='stale clue', quiz_def_source='redacted', "
                    f"ipa='tʃeɪndʒfʊl' WHERE id=%s", (word_id,))
        cur.execute(f"INSERT INTO {schema}.category (taxonomy, code, name) VALUES ('usas','A1','test cat') "
                    f"ON CONFLICT (taxonomy, code) DO NOTHING")
        cur.execute(f"SELECT id FROM {schema}.category WHERE taxonomy='usas' AND code='A1'")
        cat_id = cur.fetchone()[0]
        cur.execute(f"INSERT INTO {schema}.word_category (word_id, category_id, is_primary) VALUES (%s,%s,true)",
                    (word_id, cat_id))
        cur.execute(
            f"""INSERT INTO {schema}.word_embedding (word_id, definition_vector, definition_model, fasttext_vector, fasttext_model)
                VALUES (%s, %s, 'test-def-model', %s, 'test-ft-model')""",
            (word_id, [0.1] * 384, [0.2] * 300))
    conn.commit()

    # Re-ingesting Book One again with the SAME definition must not invalidate anything.
    same = Candidate(lemma="changeful", pos="ADJ")
    same.definition = "very susceptible to change; changing frequently"
    db.sync_book_results(conn, "Book One", kept=[same], rejected=[], schema=schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT quiz_definition FROM {schema}.word WHERE id=%s", (word_id,))
        assert cur.fetchone()[0] == "stale clue"

    # Book Two resolves "changeful" to a different, shorter sense -- this is
    # the actual trigger: definition changes on an already-enriched word.
    changed = Candidate(lemma="changeful", pos="ADJ")
    changed.definition = "Changing frequently"
    db.sync_book_results(conn, "Book Two", kept=[changed], rejected=[], schema=schema)

    with conn.cursor() as cur:
        cur.execute(f"SELECT definition, quiz_definition, quiz_def_source, ipa FROM {schema}.word WHERE id=%s",
                    (word_id,))
        defn, quiz_def, quiz_src, ipa = cur.fetchone()
        assert defn == "Changing frequently"
        assert quiz_def is None and quiz_src is None            # invalidated
        assert ipa == "tʃeɪndʒfʊl"                               # untouched -- not definition-derived

        cur.execute(f"SELECT count(*) FROM {schema}.word_category WHERE word_id=%s", (word_id,))
        assert cur.fetchone()[0] == 0                            # invalidated

        cur.execute(f"SELECT definition_vector, fasttext_vector FROM {schema}.word_embedding WHERE word_id=%s",
                    (word_id,))
        def_vec, ft_vec = cur.fetchone()
        assert def_vec is None                                   # invalidated
        assert ft_vec is not None                                # untouched -- lemma-derived, not definition-derived

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_sync_roundtrip_and_idempotent(tmp_path):
    schema = "cc_test"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    p = tmp_path / "master_vocab.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
        w.writeheader()
        for word, books in [("cangue", "BookA; BookB"), ("fuligin", "BookA")]:
            r = {c: "" for c in MASTER_COLUMNS}
            r.update(word=word, definition=f"def {word}", synonyms="x; y",
                     date_added="2026-07-05", source_book=books)
            w.writerow(r)

    s1 = db.sync_master(p, conn, schema)
    assert s1 == {"words": 2, "books": 2, "links": 3, "rows": 2}
    s2 = db.sync_master(p, conn, schema)          # idempotent
    assert s2["words"] == 2 and s2["links"] == 0  # no new links second time

    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {schema}.word"); assert cur.fetchone()[0] == 2
        cur.execute(f"select synonyms from {schema}.word where lemma='cangue'")
        assert cur.fetchone()[0] == ["x", "y"]
        cur.execute(f"""select count(*) from {schema}.word_book wb
                        join {schema}.word w on w.id=wb.word_id where w.lemma='cangue'""")
        assert cur.fetchone()[0] == 2             # linked to both books
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()


@pg
def test_junk_pos_rejection_casts_out_an_already_active_word():
    # Regression: a lemma accepted in an earlier book (e.g. its first-ever
    # dictionary lookup landed on a non-junk sense) must actually be
    # un-accepted the moment a LATER book's lookup resolves it to a proper
    # noun/symbol -- pipeline.py's post-enrichment junk-POS check now runs on
    # every re-encounter, and sync_book_results is what has to act on it.
    from concordance.model import Candidate, RejectReason

    schema = "cc_test_castout"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    accepted = Candidate(lemma="linnaea", pos="NOUN")
    accepted.definition = "a genus of plants"
    db.sync_book_results(conn, "Book One", kept=[accepted], rejected=[], schema=schema)

    with conn.cursor() as cur:
        cur.execute(f"select active from {schema}.word where lemma='linnaea'")
        assert cur.fetchone() == (True,)

    later_lookup = Candidate(lemma="linnaea", pos="NOUN")
    later_lookup.reject_reason = RejectReason.PROPER_NOUN
    later_lookup.interesting_reason = "dictionary lookup resolved this as 'proper noun' — cast out"
    stats = db.sync_book_results(conn, "Book Two", kept=[], rejected=[later_lookup], schema=schema)

    assert stats["cast_out"] == 1
    with conn.cursor() as cur:
        cur.execute(f"select active from {schema}.word where lemma='linnaea'")
        assert cur.fetchone() == (False,)
        cur.execute(f"""select reason from {schema}.rejected_word r
                        join {schema}.book b on b.id=r.book_id
                        where r.lemma='linnaea' and b.title='Book Two'""")
        assert cur.fetchone() == ("proper_noun",)

    # A junk-POS rejection for a lemma with no pre-existing word row is a
    # harmless no-op cast-out (0 rows affected), not an error.
    never_seen = Candidate(lemma="acac", pos="NOUN")
    never_seen.reject_reason = RejectReason.PROPER_NOUN
    stats2 = db.sync_book_results(conn, "Book Two", kept=[], rejected=[never_seen], schema=schema)
    assert stats2["cast_out"] == 0

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_batchable_scoring_steps_honor_limit_in_id_order():
    # normalize_word_pos/compute_archaic/compute_difficulty/compute_quizzable
    # used to have no `limit` at all (always scanned the whole table) -- now
    # that they accept one, confirm it actually caps the row count AND is
    # deterministic (ORDER BY id, not whatever order Postgres feels like
    # returning today), by seeding 5 words and checking limit=2 always
    # touches the same 2 lowest-id words, repeatably.
    from concordance.model import Candidate

    schema = "cc_test_batchable"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    words = [Candidate(lemma=f"batchword{i}", pos="noun") for i in range(5)]
    for c in words:
        c.definition = f"a definition of {c.lemma}"
    db.sync_book_results(conn, "Book One", kept=words, rejected=[], schema=schema)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {schema}.word SET part_of_speech='Noun' WHERE lemma LIKE 'batchword%%'")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(f"SELECT id, lemma FROM {schema}.word ORDER BY id")
        ordered = cur.fetchall()
    lowest_two_ids = {ordered[0][0], ordered[1][0]}

    stats = db.normalize_word_pos(conn, schema, limit=2)
    assert stats["words"] == 2
    stats_again = db.normalize_word_pos(conn, schema, limit=2)
    assert stats_again["words"] == 2  # same 2 rows every time -- deterministic, not a fluke of scan order

    dist = db.compute_archaic(conn, schema, limit=2)
    assert sum(dist.values()) == 2
    with conn.cursor() as cur:
        cur.execute(f"SELECT word_id FROM {schema}.word_difficulty")
        touched = {r[0] for r in cur.fetchall()}
    assert touched == lowest_two_ids

    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {schema}.word_difficulty")
    conn.commit()
    stats = db.compute_difficulty(conn, schema, limit=2)
    assert stats["words"] == 2
    with conn.cursor() as cur:
        cur.execute(f"SELECT word_id FROM {schema}.word_difficulty")
        touched = {r[0] for r in cur.fetchall()}
    assert touched == lowest_two_ids

    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {schema}.word_difficulty")
    conn.commit()
    dist = db.compute_quizzable(conn, schema, limit=2)
    assert sum(dist.values()) == 2
    with conn.cursor() as cur:
        cur.execute(f"SELECT word_id FROM {schema}.word_difficulty")
        touched = {r[0] for r in cur.fetchall()}
    assert touched == lowest_two_ids

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_ipa_limit_applies_after_the_only_missing_filter(monkeypatch):
    # Regression: `limit` used to slice the raw SQL fetch BEFORE the
    # only_missing filter ran in Python, so if the lowest-id rows all
    # happened to already have valid ipa, a small `limit` could return zero
    # actually-missing words even though plenty existed further down the
    # table -- the filter has to run over the full fetched set first, then
    # `limit` slices what's left.
    from concordance import wiktextract
    from concordance.model import Candidate

    monkeypatch.setattr(wiktextract, "build_lexicon", lambda *a, **k: {})

    schema = "cc_test_ipa"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    # Lowest ids (inserted first) already have valid ipa; the missing ones
    # come later in id order -- exactly the scenario the old bug mishandled.
    already_valid = [Candidate(lemma=f"validword{i}", pos="NOUN") for i in range(3)]
    still_missing = [Candidate(lemma=f"missingword{i}", pos="NOUN") for i in range(3)]
    db.sync_book_results(conn, "Book One", kept=already_valid + still_missing, rejected=[], schema=schema)
    with conn.cursor() as cur:
        for c in already_valid:
            cur.execute(f"UPDATE {schema}.word SET ipa=%s WHERE lemma=%s", ("/test/", c.lemma))
    conn.commit()

    stats = db.compute_ipa(conn, schema, limit=2)

    # The bug: with limit sliced onto the raw (ORDER-BY-less) fetch, this
    # table's 3 lowest-id rows are exactly the already-valid ones, so the
    # buggy code would see only those 2-3 rows at all -- stats["total"]
    # would come back far short of 6, and already_valid could equal total,
    # with the 3 genuinely-missing rows never even inspected.
    assert stats["total"] == 6
    assert stats["already_valid"] == 3     # unaffected by `limit`

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_ipa_sets_ipa_source_on_backfill_and_correction_and_clears_it(monkeypatch):
    from concordance import wiktextract
    from concordance.model import Candidate

    schema = "cc_test_ipa_source"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    words = [
        Candidate(lemma="freshword", pos="NOUN"),   # no existing ipa -> backfill
        Candidate(lemma="badword", pos="NOUN"),      # invalid existing ipa, kaikki has a fix -> correction
        Candidate(lemma="deadword", pos="NOUN"),     # invalid existing ipa, no fix anywhere -> cleared
    ]
    db.sync_book_results(conn, "Book One", kept=words, rejected=[], schema=schema)
    with conn.cursor() as cur:
        # French-cognate-leak pattern this sanity check exists to catch (see audio.py)
        cur.execute(f"UPDATE {schema}.word SET ipa=%s, ipa_source='wordnik' WHERE lemma=%s",
                    ("/myʁ.my.ʁe/", "badword"))
        cur.execute(f"UPDATE {schema}.word SET ipa=%s, ipa_source='wordnik' WHERE lemma=%s",
                    ("/ɑ̃.ʒe.lys/", "deadword"))
    conn.commit()

    lexicon = {
        "freshword": {"ipa": [{"ipa": "ˈfrɛʃwɜːd", "tags": ["US"]}], "audio": []},
        "badword": {"ipa": [{"ipa": "ˈbædwɜːd", "tags": ["US"]}], "audio": []},
        # deadword: no kaikki entry at all -- nothing to fix it with
    }
    monkeypatch.setattr(wiktextract, "build_lexicon", lambda *a, **k: lexicon)

    db.compute_ipa(conn, schema)

    with conn.cursor() as cur:
        cur.execute(f"SELECT lemma, ipa, ipa_source FROM {schema}.word ORDER BY lemma")
        rows = {lemma: (ipa, source) for lemma, ipa, source in cur.fetchall()}

    assert rows["freshword"] == ("ˈfrɛʃwɜːd", "kaikki")   # backfilled + sourced
    assert rows["badword"] == ("ˈbædwɜːd", "kaikki")       # corrected + sourced
    assert rows["deadword"] == (None, None)                # cleared + source cleared too

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_audio_mw_tier_downloads_when_commons_misses(monkeypatch):
    # Real recorded MW audio (mw.py's MWEntry.pronunciations[].audio_url) is
    # a same-trust-tier fallback to Commons, tried before Azure synthesis --
    # this is the tier that was sitting completely unused until now.
    from concordance import audio, mw, wiktextract
    from concordance.model import Candidate

    monkeypatch.setattr(wiktextract, "build_lexicon", lambda *a, **k: {})
    monkeypatch.setattr(audio, "azure_credentials", lambda: (None, None))
    monkeypatch.setattr(mw, "mw_api_key", lambda: "fake-key")
    monkeypatch.setattr(mw, "quota_exhausted", lambda: False)

    entry = mw.MWEntry(
        headword="besmirch", part_of_speech="verb", definitions=["to soil"],
        pronunciations=[mw.MWPronunciation(respelling="bi-SMURCH",
                                            audio_url="https://example.test/besmirch.mp3")],
        source="Merriam-Webster API",
    )
    monkeypatch.setattr(mw, "lookup_api", lambda *a, **k: [entry])
    monkeypatch.setattr(mw, "exact_matches", lambda entries, word: entries)
    monkeypatch.setattr(mw, "pick_entry", lambda entries, pos: entries[0])

    downloaded = []

    def fake_fetch(url, dest, tries=1):
        downloaded.append(url)
        return True

    monkeypatch.setattr(audio, "fetch_commons_audio", fake_fetch)

    schema = "cc_test_audio_mw"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    db.sync_book_results(conn, "Book One", kept=[Candidate(lemma="besmirch", pos="VERB")],
                          rejected=[], schema=schema)

    stats = db.compute_audio(conn, schema)

    assert stats["mw"] == 1
    assert downloaded == ["https://example.test/besmirch.mp3"]
    with conn.cursor() as cur:
        cur.execute(f"SELECT source, voice FROM {schema}.word_audio WHERE word_id = "
                    f"(SELECT id FROM {schema}.word WHERE lemma='besmirch')")
        assert cur.fetchone() == ("mw", "https://example.test/besmirch.mp3")

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_audio_skips_mw_when_quota_exhausted(monkeypatch):
    from concordance import audio, mw, wiktextract
    from concordance.model import Candidate

    monkeypatch.setattr(wiktextract, "build_lexicon", lambda *a, **k: {})
    monkeypatch.setattr(audio, "azure_credentials", lambda: (None, None))
    monkeypatch.setattr(mw, "mw_api_key", lambda: "fake-key")
    monkeypatch.setattr(mw, "quota_exhausted", lambda: True)

    called = []
    monkeypatch.setattr(mw, "lookup_api", lambda *a, **k: called.append(1) or [])

    schema = "cc_test_audio_mw_exhausted"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    db.sync_book_results(conn, "Book One", kept=[Candidate(lemma="besmirch", pos="VERB")],
                          rejected=[], schema=schema)

    stats = db.compute_audio(conn, schema)

    assert stats.get("mw", 0) == 0
    assert called == []  # never even attempted once quota was exhausted

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_backfill_ipa_from_oed(monkeypatch):
    from concordance.model import Candidate
    from concordance.oed import db as oed_db

    schema = "cc_test_oed_ipa"
    oed_schema = "oed_test_backfill"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cur.execute(f"DROP SCHEMA IF EXISTS {oed_schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    oed_db.apply_schema(conn, oed_schema)

    words = [
        Candidate(lemma="abandoner", pos="NOUN"),   # unambiguous OED match (unclosed paren in source)
        Candidate(lemma="bay", pos="NOUN"),          # homograph, but both resolved entries agree
        Candidate(lemma="fleet", pos="NOUN"),        # homograph, genuinely conflicting -- must skip
        Candidate(lemma="nomatch", pos="NOUN"),      # no oed entry at all
        Candidate(lemma="alreadyvalid", pos="NOUN"), # already has valid ipa -- must not be touched
        Candidate(lemma="feuille", pos="NOUN"),      # oed has it, but it's French -- must fail sanity check
    ]
    db.sync_book_results(conn, "Book One", kept=words, rejected=[], schema=schema)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {schema}.word SET ipa=%s, ipa_source='kaikki' WHERE lemma=%s",
                    ("/ɔːlˈrɛdi/", "alreadyvalid"))
    conn.commit()

    volume_id = oed_db.upsert_volume(conn, file_name="t.pdf", file_hash_="h1",
                                      volume_label="V", page_count=1, schema=oed_schema)

    def _entry(headword, ipa):
        entry_id = oed_db.insert_entry(
            conn, volume_id=volume_id, headword=headword, homograph_number=None,
            part_of_speech="n", etymology=None, entry_type="main", parent_entry_id=None,
            page_number=1, raw_text="raw", schema=oed_schema)
        oed_db.update_pronunciation(conn, entry_id, pronunciation_raw="raw", pass1=ipa,
                                     pass2=ipa, ipa=ipa, source="vision_llm",
                                     needs_review=False, schema=oed_schema)

    _entry("abandoner", "ˈbændənə(r")
    _entry("bay", "beɪ")
    _entry("bay", "beɪ")
    _entry("fleet", "fliːt")
    _entry("fleet", "flɛt")
    _entry("feuille", "fœj")
    conn.commit()

    stats = db.backfill_ipa_from_oed(conn, schema, oed_schema)

    with conn.cursor() as cur:
        cur.execute(f"SELECT lemma, ipa, ipa_source FROM {schema}.word ORDER BY lemma")
        rows = {lemma: (ipa, source) for lemma, ipa, source in cur.fetchall()}

    assert rows["abandoner"] == ("ˈbændənə(r)", "oed")   # paren closed, written
    assert rows["bay"] == ("beɪ", "oed")                   # homograph agreement -> written
    assert not rows["fleet"][0] and rows["fleet"][1] is None       # genuine conflict -> skipped
    assert not rows["nomatch"][0] and rows["nomatch"][1] is None   # no oed entry -> skipped
    assert rows["alreadyvalid"] == ("/ɔːlˈrɛdi/", "kaikki")  # untouched, not overridden by oed
    assert not rows["feuille"][0] and rows["feuille"][1] is None   # French IPA -> failed sanity check

    assert stats["backfilled"] == 2
    assert stats["ambiguous_homograph"] == 1
    assert stats["no_match"] == 1
    assert stats["already_valid"] == 1
    assert stats["failed_sanity_check"] == 1

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
        cur.execute(f"DROP SCHEMA {oed_schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_fill_definitions_web_tier_is_no_longer_gated_on_validity(monkeypatch, tmp_path):
    # Phase 5a: the WEB tier used to skip anything validity_score scored
    # likely-artifact, on the theory a web search for OCR noise was wasted
    # effort. That pre-gate is gone -- WEB is now tried for every word
    # nothing else defined, likely-artifact included, since that same
    # "doesn't match any dictionary" signal is exactly the rare/archaic
    # vocabulary this project's judge rubric exists to prize. validity_score
    # still runs and its estimate is still available to write for whatever
    # WEB *also* misses, just no longer used to skip the attempt.
    import llama_cpp

    from concordance import resolve, validity_score
    from concordance.model import Candidate
    from concordance.validity_score import ValidityEstimate

    class _FakeLlm:
        # fill_definitions calls .close() on whatever it loaded once the web
        # tier is done -- see its own docstring comment on why that's now
        # explicit rather than left to implicit GC timing.
        def close(self):
            pass

    monkeypatch.setattr(llama_cpp, "Llama", lambda *a, **k: _FakeLlm())
    model_path = tmp_path / "fake.gguf"
    model_path.write_bytes(b"")

    monkeypatch.setattr(resolve.localdict, "enrich", lambda cand, lex: False)
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(resolve.mw, "mw_api_key", lambda: "")
    monkeypatch.setattr(resolve.deepdef, "_from_yourdictionary", lambda cand, session: False)

    schema = "cc_test_fill"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    blank_a = Candidate(lemma="artifactword", pos="NOUN")
    blank_b = Candidate(lemma="realword", pos="NOUN")
    db.sync_book_results(conn, "Book One", kept=[blank_a, blank_b], rejected=[], schema=schema)

    def fake_estimate(word, session=None, sentence="", zipf=None):
        label = "likely-artifact" if word == "artifactword" else "plausible"
        return ValidityEstimate(word=word, score=0.0, label=label, notes="")

    monkeypatch.setattr(validity_score, "estimate", fake_estimate)

    calls = []

    def fake_web(cand, llm):
        calls.append(cand.lemma)
        # Only "realword" actually gets defined by the (fake) web search --
        # confirms the gate no longer blocks the ATTEMPT, while still
        # letting a genuine miss fall through to validity-score recording.
        if cand.lemma == "realword":
            cand.definition = f"a web definition of {cand.lemma}"
            cand.definition_source = "Web (LLM-extracted)"
            return True
        return False

    monkeypatch.setattr("concordance.websearch.define_via_web", fake_web)

    stats = db.fill_definitions(conn, schema, use_web=True, model_path=str(model_path))

    # Both words reach the web tier now -- the gate is gone.
    assert calls == ["artifactword", "realword"]
    assert stats["defined"] == 1
    assert stats["still_undefined"] == 1

    with conn.cursor() as cur:
        cur.execute(f"select definition, validity_label from {schema}.word where lemma='realword'")
        assert cur.fetchone() == ("a web definition of realword", None)
        cur.execute(f"select definition, validity_label from {schema}.word where lemma='artifactword'")
        assert cur.fetchone() == ("", "likely-artifact")

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_fill_definitions_cooldown_skips_a_recently_checked_word(monkeypatch):
    # The idempotency gap this closes: without a cooldown, every `maintain`
    # run re-attempts the ENTIRE permanently-undefined tail through
    # Wordnik/web-search again, forever. A word with a recent
    # validity_checked_at (i.e. it failed every tier recently) must be
    # skipped; one whose check is older than recheck_after_days must still
    # be retried.
    from concordance import resolve
    from concordance.model import Candidate

    monkeypatch.setattr(resolve.localdict, "enrich", lambda cand, lex: False)
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(resolve.mw, "mw_api_key", lambda: "")
    monkeypatch.setattr(resolve.deepdef, "_from_yourdictionary", lambda cand, session: False)

    schema = "cc_test_cooldown"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    recent = Candidate(lemma="recentlychecked", pos="NOUN")
    stale = Candidate(lemma="stalechecked", pos="NOUN")
    never = Candidate(lemma="neverchecked", pos="NOUN")
    db.sync_book_results(conn, "Book One", kept=[recent, stale, never], rejected=[], schema=schema)
    with conn.cursor() as cur:
        cur.execute(f"""UPDATE {schema}.word SET validity_checked_at = now() - interval '1 day'
                        WHERE lemma = 'recentlychecked'""")
        cur.execute(f"""UPDATE {schema}.word SET validity_checked_at = now() - interval '30 days'
                        WHERE lemma = 'stalechecked'""")
    conn.commit()

    stats = db.fill_definitions(conn, schema, recheck_after_days=14)

    # Only stale + never-checked are candidates; recentlychecked is skipped.
    assert stats["attempted"] == 2
    with conn.cursor() as cur:
        cur.execute(f"select validity_checked_at from {schema}.word where lemma='recentlychecked'")
        before = cur.fetchone()[0]
    assert before is not None  # untouched by this run, still the seeded value

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_fill_definitions_builds_the_lexicon_once_not_twice_per_word(monkeypatch):
    # The redundancy this whole merge eliminates: the old two-pass
    # refill-then-deepen design re-entered the cascade at Tier LOCAL twice
    # per word (once per pass). One merged pass means localdict.build_lexicon
    # runs exactly once per fill_definitions call, not once per tier/pass.
    from concordance import localdict, resolve
    from concordance.model import Candidate

    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(resolve.mw, "mw_api_key", lambda: "")
    monkeypatch.setattr(resolve.deepdef, "_from_yourdictionary", lambda cand, session: False)

    schema = "cc_test_lexicon_once"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    words = [Candidate(lemma=f"lexword{i}", pos="NOUN") for i in range(3)]
    db.sync_book_results(conn, "Book One", kept=words, rejected=[], schema=schema)

    calls = {"n": 0}
    real_build_lexicon = localdict.build_lexicon

    def spy_build_lexicon(conn_, lemmas):
        calls["n"] += 1
        return real_build_lexicon(conn_, lemmas)

    monkeypatch.setattr(localdict, "build_lexicon", spy_build_lexicon)

    db.fill_definitions(conn, schema)

    assert calls["n"] == 1  # once for the whole batch, not once per word

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_fill_definitions_checks_mw_quota_once_not_per_word(monkeypatch):
    # Same pre-check pipeline.py's ingest-time enrichment already does --
    # without it, Tier.MW would auto-discover the key regardless and burn
    # the shared 1000/day cap one word at a time on a real backlog, then
    # spend the rest of the run re-reading mw.py's on-disk cache for free.
    from concordance import mw, resolve
    from concordance.model import Candidate

    monkeypatch.setattr(resolve.localdict, "enrich", lambda cand, lex: False)
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(resolve.deepdef, "_from_yourdictionary", lambda cand, session: False)
    monkeypatch.setattr(mw, "mw_api_key", lambda: "fake-key")

    exhausted_checks = {"n": 0}

    def fake_exhausted():
        exhausted_checks["n"] += 1
        return True

    monkeypatch.setattr(mw, "quota_exhausted", fake_exhausted)

    lookup_calls = []
    monkeypatch.setattr(mw, "lookup_api", lambda *a, **k: lookup_calls.append(1) or [])

    schema = "cc_test_mw_quota"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    words = [Candidate(lemma=f"quotaword{i}", pos="NOUN") for i in range(3)]
    db.sync_book_results(conn, "Book One", kept=words, rejected=[], schema=schema)

    db.fill_definitions(conn, schema)

    assert exhausted_checks["n"] == 1     # checked once for the whole batch
    assert lookup_calls == []             # MW never actually called once exhausted

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_sync_book_results_writes_the_variant_review_flag():
    # Candidate.variant_flag_reason/_note (set by pipeline.py when
    # validity_score.variant_reject_reason fires on a KEPT word) must reach
    # word.variant_flag_reason/_note/_at -- the human-review queue, not an
    # auto-reject: the word stays active and defined either way.
    from concordance.model import Candidate

    schema = "cc_test_variant_flag"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    flagged = Candidate(lemma="acte", pos="NOUN")
    flagged.definition = "a specific action or deed"
    flagged.variant_flag_reason = "foreign_language"
    flagged.variant_flag_note = "looks fr (zipf 4.8 there vs English)"
    unflagged = Candidate(lemma="armiger", pos="NOUN")
    unflagged.definition = "a person entitled to bear heraldic arms"

    db.sync_book_results(conn, "Book One", kept=[flagged, unflagged], rejected=[], schema=schema)

    with conn.cursor() as cur:
        cur.execute(f"""select active, definition, variant_flag_reason, variant_flag_note,
                                variant_flagged_at is not null
                         from {schema}.word where lemma='acte'""")
        assert cur.fetchone() == (True, "a specific action or deed", "foreign_language",
                                   "looks fr (zipf 4.8 there vs English)", True)
        cur.execute(f"""select variant_flag_reason, variant_flagged_at
                         from {schema}.word where lemma='armiger'""")
        assert cur.fetchone() == (None, None)

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_fill_definitions_flags_but_does_not_cast_out_a_variant_hit(monkeypatch):
    # The reverted design: a foreign word / archaic-spelling variant that a
    # source successfully defines gets FLAGGED for human review, not cast
    # out -- stays active=true, defined normally, distinguishable only via
    # variant_flag_reason.
    from concordance import resolve
    from concordance.model import Candidate

    monkeypatch.setattr(resolve.localdict, "enrich", lambda cand, lex: False)

    def fake_freedict(cand, session):
        if cand.lemma == "acte":
            cand.definition = "a specific action or deed"
            cand.definition_source = "Free Dictionary API"
            cand.part_of_speech = "noun"
        elif cand.lemma == "armiger":
            cand.definition = "a person entitled to bear heraldic arms"
            cand.definition_source = "Free Dictionary API"
            cand.part_of_speech = "noun"

    monkeypatch.setattr(resolve.dictionary, "enrich", fake_freedict)

    schema = "cc_test_variant_flag_fill"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    words = [Candidate(lemma=l, pos="NOUN") for l in ("acte", "armiger")]
    db.sync_book_results(conn, "Book One", kept=words, rejected=[], schema=schema)

    stats = db.fill_definitions(conn, schema)

    assert stats["cast_out"] == 0
    assert stats["defined"] == 2   # both accepted -- neither cast out

    with conn.cursor() as cur:
        cur.execute(f"select active, variant_flag_reason from {schema}.word where lemma='acte'")
        assert cur.fetchone() == (True, "foreign_language")
        cur.execute(f"select active, variant_flag_reason from {schema}.word where lemma='armiger'")
        assert cur.fetchone() == (True, None)

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_fill_definitions_reaches_oed_tier_and_writes_the_review_flag(monkeypatch):
    # Regression test for the variant_flag merge bug: fill_definitions used
    # to thread ONLY validity_score.variant_reject_reason's result through
    # the UPDATE's COALESCE, silently dropping cand.variant_flag_reason (set
    # by resolve.py's Tier.OED to mark an OED-sourced definition for human
    # review) whenever that local check found nothing -- which is the common
    # case, since a real OED hit is neither a foreign word nor a misspelling.
    from concordance import resolve
    from concordance.model import Candidate
    from concordance.oed import db as oed_db

    monkeypatch.setattr(resolve.localdict, "enrich", lambda cand, lex: False)
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)

    schema = "cc_test_oed_tier_fill"
    oed_schema = "oed_test_tier_fill"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cur.execute(f"DROP SCHEMA IF EXISTS {oed_schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    oed_db.apply_schema(conn, oed_schema)

    volume_id = oed_db.upsert_volume(conn, file_name="test.pdf", file_hash_="fill-defs-test",
                                      volume_label="Test", page_count=1, schema=oed_schema)
    entry_id = oed_db.insert_entry(
        conn, volume_id=volume_id, headword="tetarteron", homograph_number=None,
        part_of_speech="sb", etymology=None, entry_type="main", parent_entry_id=None,
        page_number=1, raw_text="raw", schema=oed_schema)
    oed_db.insert_definitions(conn, entry_id, [
        {"sense_label": None, "definition_text": "A Byzantine gold coin of the 10th-11th centuries."},
    ], schema=oed_schema)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {oed_schema}.entry SET lemma=true WHERE id=%s", (entry_id,))
    conn.commit()

    words = [Candidate(lemma="tetarteron", pos="NOUN")]
    db.sync_book_results(conn, "Book One", kept=words, rejected=[], schema=schema)

    stats = db.fill_definitions(conn, schema, oed_schema=oed_schema)

    assert stats["defined"] == 1
    with conn.cursor() as cur:
        cur.execute(f"select definition_source, variant_flag_reason from {schema}.word "
                    f"where lemma='tetarteron'")
        source, flag = cur.fetchone()
        assert source == "OED"
        assert flag == "oed_unverified"

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
        cur.execute(f"DROP SCHEMA {oed_schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_mw_backfill_defines_skips_and_casts_out(monkeypatch):
    # One candidate per outcome: a real MW hit (defined), a genuine miss (no
    # MW entry), and an MW hit that resolves to a leaked proper noun (cast
    # out) -- confirmed live during development that MW's own category names
    # ("biographical name", "geographical name", "trademark") weren't yet
    # recognized by the shared junk_pos_reason check, silently letting real
    # proper nouns (a former Canadian PM, a French colonial territory) land
    # in the accepted list with a genuine, correctly-sourced definition.
    from concordance import mw as mw_module
    from concordance.model import Candidate

    def fake_lookup_api(word, api_key, session=None, console=None):
        if word == "realword":
            return [mw_module.MWEntry(headword="realword", part_of_speech="noun",
                                       definitions=["a genuine definition"],
                                       etymology="from Old English", first_known_use="14th century")]
        if word == "bioword":
            return [mw_module.MWEntry(headword="bioword", part_of_speech="biographical name",
                                       definitions=["Someone 1900-1980, a person"])]
        return []  # "missingword" -- MW has no exact entry at all

    monkeypatch.setattr(mw_module, "lookup_api", fake_lookup_api)
    monkeypatch.setattr(mw_module, "mw_api_key", lambda: "fake-key")

    schema = "cc_test_mw_backfill"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    words = [Candidate(lemma=l, pos="NOUN") for l in ("realword", "missingword", "bioword")]
    db.sync_book_results(conn, "Book One", kept=words, rejected=[], schema=schema)
    # A fourth word already scored likely-artifact must never be a candidate
    # at all -- mw_backfill only targets null/uncertain/likely-valid.
    artifact = Candidate(lemma="artifactword", pos="NOUN")
    db.sync_book_results(conn, "Book One", kept=[artifact], rejected=[], schema=schema)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {schema}.word SET validity_label='likely-artifact' WHERE lemma='artifactword'")
    conn.commit()

    stats = db.mw_backfill(conn, schema, use_scrape=False)

    assert stats["attempted"] == 3   # NOT artifactword
    assert stats["defined"] == 1
    assert stats["no_entry"] == 1
    assert stats["cast_out"] == 1

    with conn.cursor() as cur:
        cur.execute(f"""select definition, definition_source, etymology, first_known_use,
                                active, mw_checked_at is not null
                        from {schema}.word where lemma='realword'""")
        assert cur.fetchone() == ("a genuine definition", "Merriam-Webster API",
                                   "from Old English", "14th century", True, True)

        cur.execute(f"select definition, active, mw_checked_at is not null "
                    f"from {schema}.word where lemma='missingword'")
        assert cur.fetchone() == ("", True, True)

        cur.execute(f"select active, mw_checked_at is not null from {schema}.word where lemma='bioword'")
        assert cur.fetchone() == (False, True)

        cur.execute(f"select mw_checked_at from {schema}.word where lemma='artifactword'")
        assert cur.fetchone() == (None,)

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_mw_backfill_never_overwrites_existing_metadata_with_blank(monkeypatch):
    # Regression: an early version's COALESCE was backwards (kept the OLD
    # column value whenever it was non-blank, rather than preferring the NEW
    # one) -- caught live when a word's stale, unrelated definition_source
    # ('dictionary', legacy data with no matching definition) silently
    # survived a real MW-sourced definition instead of being replaced with
    # 'Merriam-Webster API'. This confirms the fixed direction: MW's new,
    # non-blank value wins; an existing value is kept ONLY where MW's own is
    # blank (etymology here).
    from concordance import mw as mw_module
    from concordance.model import Candidate

    monkeypatch.setattr(mw_module, "lookup_api", lambda word, api_key, session=None, console=None: [
        mw_module.MWEntry(headword="staleword", part_of_speech="noun",
                           definitions=["a fresh MW definition"], etymology=""),
    ])
    monkeypatch.setattr(mw_module, "mw_api_key", lambda: "fake-key")

    schema = "cc_test_mw_backfill_coalesce"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    words = [Candidate(lemma="staleword", pos="NOUN")]
    db.sync_book_results(conn, "Book One", kept=words, rejected=[], schema=schema)
    with conn.cursor() as cur:
        # Simulates real observed legacy data: a non-blank definition_source/
        # etymology with no matching definition (definition stays blank, so
        # this row is still a valid mw_backfill candidate).
        cur.execute(f"""UPDATE {schema}.word SET definition_source='dictionary',
                        etymology='a pre-existing etymology' WHERE lemma='staleword'""")
    conn.commit()

    db.mw_backfill(conn, schema, use_scrape=False)

    with conn.cursor() as cur:
        cur.execute(f"select definition, definition_source, etymology from {schema}.word where lemma='staleword'")
        # definition_source: MW's new value overwrites the stale 'dictionary' tag.
        # etymology: MW supplied none, so the pre-existing value survives.
        assert cur.fetchone() == ("a fresh MW definition", "Merriam-Webster API", "a pre-existing etymology")

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_author_fame_excludes_placeholders_and_is_resumable(monkeypatch):
    from concordance import fame

    monkeypatch.setattr(fame, "gather_author_evidence", lambda author, session: {"ngram": {}, "wikidata": {}})
    monkeypatch.setattr(fame, "score_author", lambda llm, author, factors: (7.0, "well known"))

    schema = "cc_test_author_fame"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    db.sync_book_results(conn, "A Real Book", kept=[], rejected=[], schema=schema, author="A Real Author")
    db.sync_book_results(conn, "An Anthology", kept=[], rejected=[], schema=schema, author="Various")

    stats = db.compute_author_fame(conn, schema, llm=object())

    # "Various" (a PLACEHOLDER_AUTHORS aggregation label, not a person) must
    # never be scored -- it has no individual fame to speak of, and scoring
    # it would burn a real LLM call + network round-trips on nothing.
    assert stats["attempted"] == 1
    with conn.cursor() as cur:
        cur.execute(f"select fame_score, fame_reasoning, computed_at is not null, checked_at is not null "
                    f"from {schema}.author_fame where author='A Real Author'")
        assert cur.fetchone() == (7.0, "well known", True, True)
        cur.execute(f"select count(*) from {schema}.author_fame where author='Various'")
        assert cur.fetchone() == (0,)

    # Resumability: a second run with the default stale_days=0 must touch
    # nothing already-scored -- checked_at is sticky, same convention as
    # word.mw_checked_at.
    monkeypatch.setattr(fame, "score_author", lambda llm, author, factors: (1.0, "should not be written"))
    stats2 = db.compute_author_fame(conn, schema, llm=object())
    assert stats2["attempted"] == 0
    with conn.cursor() as cur:
        cur.execute(f"select fame_score from {schema}.author_fame where author='A Real Author'")
        assert cur.fetchone() == (7.0,)   # untouched, not overwritten with 1.0

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_book_fame_uses_author_context_and_tolerates_its_absence(monkeypatch):
    from concordance import fame

    captured_author_fame = {}

    def fake_gather_book_evidence(title, author, author_fame, session):
        captured_author_fame[title] = author_fame
        return {"ngram": {"skipped": True}, "author_fame_seen": author_fame}

    monkeypatch.setattr(fame, "gather_book_evidence", fake_gather_book_evidence)
    monkeypatch.setattr(fame, "score_book", lambda llm, title, author, factors: (4.0, "unremarkable"))

    schema = "cc_test_book_fame"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    db.sync_book_results(conn, "A Famous Author's Book", kept=[], rejected=[], schema=schema, author="Famous Author")
    db.sync_book_results(conn, "An Unscored Author's Book", kept=[], rejected=[], schema=schema, author="Unscored Author")
    with conn.cursor() as cur:
        cur.execute(f"""INSERT INTO {schema}.author_fame (author, fame_score, fame_reasoning, computed_at, checked_at)
                        VALUES ('Famous Author', 9.0, 'major figure', now(), now())""")
    conn.commit()

    stats = db.compute_book_fame(conn, schema, llm=object())
    assert stats["attempted"] == 2
    assert stats["scored"] == 2

    # The famous author's book got their real score+reasoning as context...
    assert captured_author_fame["A Famous Author's Book"]["fame_score"] == 9.0
    assert captured_author_fame["A Famous Author's Book"]["fame_reasoning"] == "major figure"
    # ...while the not-yet-scored author's book saw None, not a crash or a
    # missing key -- book-fame must tolerate no prior existing at all.
    assert captured_author_fame["An Unscored Author's Book"] is None

    with conn.cursor() as cur:
        cur.execute(f"""select bf.fame_factors->'author_fame_seen'->>'fame_score'
                        from {schema}.book_fame bf join {schema}.book b on b.id=bf.book_id
                        where b.title='A Famous Author''s Book'""")
        assert cur.fetchone() == ("9.0",)

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_author_fame_one_failure_does_not_abort_the_run(monkeypatch):
    from concordance import fame

    def flaky_gather(author, session):
        if author == "Poison Author":
            raise RuntimeError("simulated evidence-gathering crash")
        return {"ngram": {}, "wikidata": {}}

    monkeypatch.setattr(fame, "gather_author_evidence", flaky_gather)
    monkeypatch.setattr(fame, "score_author", lambda llm, author, factors: (5.0, "fine"))

    schema = "cc_test_author_fame_flaky"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    for author in ("Author Alpha", "Poison Author", "Author Beta"):
        db.sync_book_results(conn, f"{author}'s Book", kept=[], rejected=[], schema=schema, author=author)

    stats = db.compute_author_fame(conn, schema, llm=object())

    # The poisoned author is skipped (not scored), but its neighbors in the
    # same run still get processed -- one bad item must not take the whole
    # run down.
    with conn.cursor() as cur:
        cur.execute(f"select author, fame_score from {schema}.author_fame order by author")
        rows = dict(cur.fetchall())
    assert rows.get("Author Alpha") == 5.0
    assert rows.get("Author Beta") == 5.0
    assert "Poison Author" not in rows
    assert stats["attempted"] == 2   # only the two that didn't raise

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_author_fame_stops_early_on_evidence_degradation(monkeypatch):
    from concordance import db as dbmod
    from concordance import fame

    monkeypatch.setattr(dbmod, "_FAME_EVIDENCE_FAILURE_MIN_SAMPLE", 3)
    monkeypatch.setattr(fame, "gather_author_evidence",
                        lambda author, session: {"ngram": {"failed": True}, "wikidata": {"failed": True}, "snippets_failed": True})
    monkeypatch.setattr(fame, "score_author", lambda llm, author, factors: (1.0, "blind guess"))

    schema = "cc_test_author_fame_degraded"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    for i in range(10):
        db.sync_book_results(conn, f"Book {i}", kept=[], rejected=[], schema=schema, author=f"Author {i}")

    stats = db.compute_author_fame(conn, schema, llm=object())

    assert stats["stopped_early"] is True
    assert stats["remaining"] > 0
    assert stats["attempted"] < 10   # the run genuinely stopped, not just labeled

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_merge_book_group_repoints_dedupes_and_deletes():
    from concordance.model import Candidate, RejectReason

    schema = "cc_test_book_merge"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    # Part 1: keeps "alpha" (shared with part 2) and "solo1"; rejects "junk" (shared with part 3).
    part1_kept = [Candidate(lemma="alpha", pos="NOUN"), Candidate(lemma="solo1", pos="NOUN")]
    for c in part1_kept:
        c.definition = f"a definition of {c.lemma}"
    junk1 = Candidate(lemma="junk", pos="NOUN")
    junk1.reject_reason = RejectReason.NOT_INTERESTING
    db.sync_book_results(conn, "Some Book, Volume 1", kept=part1_kept, rejected=[junk1],
                        schema=schema, author="Some Author")

    # Part 2: keeps "alpha" (again -- must not violate word_book's PK on repoint) and "solo2".
    part2_kept = [Candidate(lemma="alpha", pos="NOUN"), Candidate(lemma="solo2", pos="NOUN")]
    for c in part2_kept:
        c.definition = f"a definition of {c.lemma}"
    db.sync_book_results(conn, "Some Book, Volume 2", kept=part2_kept, rejected=[],
                        schema=schema, author="Some Author")

    # Part 3: rejects "junk" again (must not violate rejected_word's UNIQUE(book_id,lemma_lc) on repoint).
    junk3 = Candidate(lemma="junk", pos="NOUN")
    junk3.reject_reason = RejectReason.NOT_INTERESTING
    db.sync_book_results(conn, "Some Book, Volume 3", kept=[], rejected=[junk3],
                        schema=schema, author="Some Author")

    with conn.cursor() as cur:
        cur.execute(f"SELECT id FROM {schema}.book WHERE title LIKE 'Some Book, Volume%' ORDER BY title")
        ids = [r[0] for r in cur.fetchall()]
    survivor, others = ids[0], ids[1:]

    # Seed precomputed/derived rows for every id in the group, both
    # book_similarity directions, to confirm they're ALL cleared, not just
    # the non-survivor ones.
    with conn.cursor() as cur:
        for a, b in [(ids[0], ids[1]), (ids[1], ids[0]), (ids[0], ids[2])]:
            cur.execute(f"""INSERT INTO {schema}.book_similarity (book_a_id, book_b_id, score, shared_word_count)
                            VALUES (%s,%s,0.5,3)""", (a, b))
        for bid in ids:
            cur.execute(f"""INSERT INTO {schema}.book_cluster (book_id, title, author, cluster_id, mds_x, mds_y, word_count)
                            VALUES (%s,'x','Some Author',1,0.0,0.0,1)""", (bid,))
            cur.execute(f"""INSERT INTO {schema}.book_fame (book_id, fame_score, checked_at)
                            VALUES (%s, 5, now())""", (bid,))
    conn.commit()

    stats = db.merge_book_group(
        conn, schema, survivor, others,
        title="Some Book (Complete)", author="Some Author", archive_path="archive/Some Book (Complete) -- Some Author.txt",
        word_count=1000, distinct_nonstop_word_count=400)
    assert stats == {"survivor_book_id": survivor, "merged_count": 2}

    with conn.cursor() as cur:
        # word_book: survivor has alpha, solo1, solo2 -- no PK violation, no duplicate row for alpha.
        cur.execute(f"""SELECT w.lemma FROM {schema}.word_book wb JOIN {schema}.word w ON w.id=wb.word_id
                        WHERE wb.book_id=%s ORDER BY w.lemma""", (survivor,))
        assert [r[0] for r in cur.fetchall()] == ["alpha", "solo1", "solo2"]

        # rejected_word: exactly one row for "junk" on the survivor, not two.
        cur.execute(f"SELECT lemma FROM {schema}.rejected_word WHERE book_id=%s", (survivor,))
        assert [r[0] for r in cur.fetchall()] == ["junk"]

        # precomputed/derived tables cleared for the WHOLE group, survivor included.
        cur.execute(f"SELECT count(*) FROM {schema}.book_similarity WHERE book_a_id = ANY(%s) OR book_b_id = ANY(%s)",
                    (ids, ids))
        assert cur.fetchone() == (0,)
        cur.execute(f"SELECT count(*) FROM {schema}.book_cluster WHERE book_id = ANY(%s)", (ids,))
        assert cur.fetchone() == (0,)
        cur.execute(f"SELECT count(*) FROM {schema}.book_fame WHERE book_id = ANY(%s)", (ids,))
        assert cur.fetchone() == (0,)

        # non-survivor book rows are gone; survivor reflects the compiled values.
        cur.execute(f"SELECT count(*) FROM {schema}.book WHERE id = ANY(%s)", (others,))
        assert cur.fetchone() == (0,)
        cur.execute(f"""SELECT title, author, archive_path, word_count, distinct_nonstop_word_count
                        FROM {schema}.book WHERE id=%s""", (survivor,))
        assert cur.fetchone() == ("Some Book (Complete)", "Some Author",
                                    "archive/Some Book (Complete) -- Some Author.txt", 1000, 400)

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_merge_book_group_is_idempotent_on_rerun():
    schema = "cc_test_book_merge_idempotent"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    from concordance.model import Candidate
    cand = Candidate(lemma="solo", pos="NOUN")
    cand.definition = "a definition"
    db.sync_book_results(conn, "Solo Book", kept=[cand], rejected=[], schema=schema, author="Some Author")
    with conn.cursor() as cur:
        cur.execute(f"SELECT id FROM {schema}.book WHERE title='Solo Book'")
        survivor = cur.fetchone()[0]

    # other_book_ids that never existed at all -- a rerun after a successful
    # merge (or a merge whose "other" side was already cleaned up some other
    # way) must be a no-op, not an error.
    stats = db.merge_book_group(
        conn, schema, survivor, [999999, 999998],
        title="Solo Book (Complete)", author="Some Author",
        archive_path="archive/Solo Book (Complete) -- Some Author.txt",
        word_count=10, distinct_nonstop_word_count=5)
    assert stats == {"survivor_book_id": survivor, "merged_count": 2}

    with conn.cursor() as cur:
        cur.execute(f"SELECT title FROM {schema}.book WHERE id=%s", (survivor,))
        assert cur.fetchone() == ("Solo Book (Complete)",)

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_book_merge_manifest_upsert_preserves_terminal_timestamps():
    # The book_merge_group manifest's core resumability contract:
    # compiled_at/merged_at are the ONLY terminal markers -- everything
    # else (part_book_ids, skip_reason, checked_at) is meant to be
    # overwritten on every re-detection, since eligibility can change as
    # the corpus changes (e.g. a gap closing once a volume is ingested).
    schema = "cc_test_book_merge_manifest"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    group_id = db.upsert_book_merge_group(
        conn, schema, title_base="Some Book", author="Some Author",
        part_book_ids=[1, 2], part_labels=[{"book_id": 1, "num1": 1, "num2": 1},
                                            {"book_id": 2, "num1": 2, "num2": 2}],
        survivor_book_id=1, skip_reason=None, gap_detail=None)
    conn.commit()
    db.mark_book_merge_compiled(conn, schema, group_id, "archive/Some Book (Complete) -- Some Author.txt")

    # Re-detection runs again (e.g. the CLI's next invocation) and re-upserts
    # the SAME group -- compiled_at must survive this, not be wiped back to NULL.
    group_id_2 = db.upsert_book_merge_group(
        conn, schema, title_base="Some Book", author="Some Author",
        part_book_ids=[1, 2], part_labels=[{"book_id": 1, "num1": 1, "num2": 1},
                                            {"book_id": 2, "num1": 2, "num2": 2}],
        survivor_book_id=1, skip_reason=None, gap_detail=None)
    assert group_id_2 == group_id   # same (title_base, author) -- same manifest row

    with conn.cursor() as cur:
        cur.execute(f"SELECT compiled_at IS NOT NULL, merged_at IS NOT NULL, compiled_path "
                    f"FROM {schema}.book_merge_group WHERE id=%s", (group_id,))
        assert cur.fetchone() == (True, False, "archive/Some Book (Complete) -- Some Author.txt")

    db.mark_book_merge_merged(conn, schema, group_id)
    with conn.cursor() as cur:
        cur.execute(f"SELECT merged_at IS NOT NULL FROM {schema}.book_merge_group WHERE id=%s", (group_id,))
        assert cur.fetchone() == (True,)

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_dedupe_plural_definitions_all_three_outcomes(monkeypatch):
    from concordance import resolve
    from concordance.model import Candidate

    monkeypatch.setattr(resolve.localdict, "enrich", lambda cand, lex: False)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(resolve.mw, "mw_api_key", lambda: "")
    monkeypatch.setattr(resolve.deepdef, "_from_yourdictionary", lambda cand, session: False)

    def fake_freedict(cand, session):
        if cand.lemma == "goblin":
            cand.definition = "A grotesque, mischievous creature of folklore."
            cand.definition_source = "Free Dictionary API"
            cand.part_of_speech = "noun"
        elif cand.lemma == "quisling":
            cand.definition = "A traitor who collaborates with an enemy occupying force."
            cand.definition_source = "Free Dictionary API"
            cand.part_of_speech = "proper noun"  # deliberately junk -- should cast out

    monkeypatch.setattr(resolve.dictionary, "enrich", fake_freedict)

    schema = "cc_test_dedupe_plurals"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    # Case 1: "linked" -- singular already active.
    fairy = Candidate(lemma="fairy", pos="NOUN")
    fairy.definition = "A mythical being with magical powers."
    fairies = Candidate(lemma="fairies", pos="NOUN")
    fairies.definition = "plural of fairy"

    # Case 2: "left_inactive" -- singular exists but is inactive (deliberately pruned).
    troll = Candidate(lemma="troll", pos="NOUN")
    troll.definition = "A cave-dwelling creature of folklore."
    trolls = Candidate(lemma="trolls", pos="NOUN")
    trolls.definition = "plural of troll"

    # Case 3a: "created" -- singular doesn't exist, resolves cleanly.
    goblins = Candidate(lemma="goblins", pos="NOUN")
    goblins.definition = "Plural of goblin."

    # Case 3b: "created" -> "cast_out" -- singular doesn't exist, resolves to junk POS.
    quislings = Candidate(lemma="quislings", pos="NOUN")
    quislings.definition = "Plural of quisling."

    db.sync_book_results(conn, "Book One",
                         kept=[fairy, fairies, troll, trolls, goblins, quislings],
                         rejected=[], schema=schema)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {schema}.word SET active=false WHERE lemma='troll'")
    conn.commit()

    stats = db.dedupe_plural_definitions(conn, schema, use_web=False)

    assert stats["attempted"] == 4  # fairies, trolls, goblins, quislings
    assert stats["linked"] == 1
    assert stats["left_inactive"] == 1
    assert stats["created"] == 1
    assert stats["cast_out"] == 1

    with conn.cursor() as cur:
        # fairy: untouched and still active; fairies: deactivated.
        cur.execute(f"select active from {schema}.word where lemma='fairy'")
        assert cur.fetchone() == (True,)
        cur.execute(f"select active from {schema}.word where lemma='fairies'")
        assert cur.fetchone() == (False,)

        # troll: still inactive (not resurrected); trolls: also deactivated.
        cur.execute(f"select active from {schema}.word where lemma='troll'")
        assert cur.fetchone() == (False,)
        cur.execute(f"select active from {schema}.word where lemma='trolls'")
        assert cur.fetchone() == (False,)

        # goblin: newly created, active, properly defined; goblins: deactivated.
        cur.execute(f"select active, definition from {schema}.word where lemma='goblin'")
        assert cur.fetchone() == (True, "A grotesque, mischievous creature of folklore.")
        cur.execute(f"select active from {schema}.word where lemma='goblins'")
        assert cur.fetchone() == (False,)

        # quisling: created but cast out (junk POS); quislings: also deactivated.
        cur.execute(f"select active, part_of_speech from {schema}.word where lemma='quisling'")
        assert cur.fetchone() == (False, "proper noun")
        cur.execute(f"select active from {schema}.word where lemma='quislings'")
        assert cur.fetchone() == (False,)

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_dedupe_plural_definitions_is_idempotent(monkeypatch):
    # A plural already deactivated by an earlier run must not be reselected.
    from concordance import resolve
    from concordance.model import Candidate

    monkeypatch.setattr(resolve.localdict, "enrich", lambda cand, lex: False)
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(resolve.mw, "mw_api_key", lambda: "")
    monkeypatch.setattr(resolve.deepdef, "_from_yourdictionary", lambda cand, session: False)

    schema = "cc_test_dedupe_idempotent"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    imp = Candidate(lemma="imp", pos="NOUN")
    imp.definition = "A small, mischievous devil."
    imps = Candidate(lemma="imps", pos="NOUN")
    imps.definition = "plural of imp"
    db.sync_book_results(conn, "Book One", kept=[imp, imps], rejected=[], schema=schema)

    stats1 = db.dedupe_plural_definitions(conn, schema, use_web=False)
    assert stats1["attempted"] == 1 and stats1["linked"] == 1

    stats2 = db.dedupe_plural_definitions(conn, schema, use_web=False)
    assert stats2["attempted"] == 0  # imps is already inactive -- not reselected

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_expand_synonym_definitions_all_outcomes(monkeypatch):
    from concordance import resolve
    from concordance.model import Candidate

    monkeypatch.setattr(resolve.localdict, "enrich", lambda cand, lex: False)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(resolve.mw, "mw_api_key", lambda: "")
    monkeypatch.setattr(resolve.deepdef, "_from_yourdictionary", lambda cand, session: False)

    def fake_freedict(cand, session):
        if cand.lemma == "grotesque":
            cand.definition = "A fantastically distorted or ugly figure or creature."
            cand.definition_source = "Free Dictionary API"
            cand.part_of_speech = "noun"
        elif cand.lemma == "quisling":
            cand.definition = "A traitor who collaborates with an occupying enemy force."
            cand.definition_source = "Free Dictionary API"
            cand.part_of_speech = "proper noun"  # deliberately junk

    monkeypatch.setattr(resolve.dictionary, "enrich", fake_freedict)

    schema = "cc_test_expand_synonyms"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    words = []

    # Case 1: embedded quoted gloss -- extracted directly, no lookup.
    w1 = Candidate(lemma="niddering", pos="NOUN")
    w1.definition = 'Synonym of nithing ("a coward, a dastard; a wretch").'
    words.append(w1)

    # Case 2: real content on a later line.
    w2 = Candidate(lemma="assoilzie", pos="VERB")
    w2.definition = "Synonym of assoil.\nTo absolve or release (someone) from blame or sin."
    words.append(w2)

    # Case 3: bare, target already active with a real definition -- reused.
    target_active = Candidate(lemma="lapidary", pos="ADJ")
    target_active.definition = "Relating to the engraving of gemstones."
    w3 = Candidate(lemma="lapidarian", pos="ADJ")
    w3.definition = "Synonym of lapidary."
    words.extend([target_active, w3])

    # Case 4: bare, target exists but inactive -- left unchanged.
    target_inactive = Candidate(lemma="unadvisedly", pos="ADV")
    target_inactive.definition = "In an unadvised manner."
    w4 = Candidate(lemma="inadvisedly", pos="ADV")
    w4.definition = "Synonym of unadvisedly."
    words.extend([target_inactive, w4])

    # Case 5: bare, target doesn't exist -- resolved and created cleanly.
    w5 = Candidate(lemma="goblinesque", pos="ADJ")
    w5.definition = "Synonym of grotesque."
    words.append(w5)

    # Case 6: bare, target doesn't exist -- resolves to junk POS, cast out.
    w6 = Candidate(lemma="fifthcolumnist", pos="NOUN")
    w6.definition = "Synonym of quisling."
    words.append(w6)

    db.sync_book_results(conn, "Book One", kept=words, rejected=[], schema=schema)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {schema}.word SET active=false WHERE lemma='unadvisedly'")
    conn.commit()

    stats = db.expand_synonym_definitions(conn, schema, use_web=False)

    assert stats["attempted"] == 6
    assert stats["extracted"] == 2       # niddering, assoilzie
    assert stats["reused_existing"] == 1  # lapidarian
    assert stats["target_inactive"] == 1  # inadvisedly
    assert stats["target_created"] == 1   # goblinesque
    assert stats["target_cast_out"] == 1  # fifthcolumnist

    with conn.cursor() as cur:
        cur.execute(f"select definition from {schema}.word where lemma='niddering'")
        assert cur.fetchone() == ("a coward, a dastard; a wretch",)

        cur.execute(f"select definition from {schema}.word where lemma='assoilzie'")
        assert cur.fetchone() == ("To absolve or release (someone) from blame or sin.",)

        cur.execute(f"select definition from {schema}.word where lemma='lapidarian'")
        assert cur.fetchone() == ("Relating to the engraving of gemstones.",)

        # inadvisedly left unchanged -- target is inactive.
        cur.execute(f"select definition from {schema}.word where lemma='inadvisedly'")
        assert cur.fetchone() == ("Synonym of unadvisedly.",)
        cur.execute(f"select active from {schema}.word where lemma='unadvisedly'")
        assert cur.fetchone() == (False,)  # not reactivated

        # goblinesque upgraded; grotesque created active with the real definition.
        cur.execute(f"select definition from {schema}.word where lemma='goblinesque'")
        assert cur.fetchone() == ("A fantastically distorted or ugly figure or creature.",)
        cur.execute(f"select active, definition from {schema}.word where lemma='grotesque'")
        assert cur.fetchone() == (True, "A fantastically distorted or ugly figure or creature.")

        # fifthcolumnist left unchanged; quisling created but cast out.
        cur.execute(f"select definition from {schema}.word where lemma='fifthcolumnist'")
        assert cur.fetchone() == ("Synonym of quisling.",)
        cur.execute(f"select active, part_of_speech from {schema}.word where lemma='quisling'")
        assert cur.fetchone() == (False, "proper noun")

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_expand_synonym_definitions_strips_css_junk_and_is_idempotent(monkeypatch):
    # "idiocy" is pre-seeded as an active, defined word so this hits the
    # reused_existing branch (no lookup/network needed) -- the point of this
    # test is the CSS-junk stripping and idempotency, not resolution.
    from concordance.model import Candidate

    schema = "cc_test_expand_synonyms_css"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    idiocy = Candidate(lemma="idiocy", pos="NOUN")
    idiocy.definition = "Extremely foolish behaviour."
    w = Candidate(lemma="idiotcy", pos="NOUN")
    w.definition = "Synonym of idiocy. .mw-parser-output .defdate{font-size:smaller}"
    db.sync_book_results(conn, "Book One", kept=[idiocy, w], rejected=[], schema=schema)

    stats = db.expand_synonym_definitions(conn, schema, use_web=False)
    assert stats["attempted"] == 1
    assert stats["reused_existing"] == 1

    with conn.cursor() as cur:
        cur.execute(f"select definition from {schema}.word where lemma='idiotcy'")
        defn = cur.fetchone()[0]
        assert ".mw-parser-output" not in defn
        assert defn == "Extremely foolish behaviour."

    # Second run: nothing left to do -- "synonym of" no longer appears anywhere.
    stats2 = db.expand_synonym_definitions(conn, schema, use_web=False)
    assert stats2["attempted"] == 0

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_definition_links_matches_lemma_and_excludes_self_and_inactive():
    from concordance.model import Candidate

    schema = "cc_test_definition_links"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    # recondite -> abstruse: exact-lemma mention, real cross-link expected.
    abstruse = Candidate(lemma="abstruse", pos="ADJ")
    abstruse.definition = "Difficult to understand; obscure."
    recondite = Candidate(lemma="recondite", pos="ADJ")
    recondite.definition = "Abstruse and obscure in nature."

    # vilify -> mentioned via an INFLECTED form ("vilified") -- tests
    # lemma-aware matching, not just exact substring matching.
    vilify = Candidate(lemma="vilify", pos="VERB")
    vilify.definition = "To speak or write about in an abusively disparaging manner."
    banisher = Candidate(lemma="banisher", pos="NOUN")
    banisher.definition = "One who has vilified or exiled another."

    # amble: a word whose own definition happens to reuse its own lemma --
    # must NOT create a self-link.
    amble = Candidate(lemma="amble", pos="VERB")
    amble.definition = "To amble along at a leisurely pace."

    # inert: target exists but is INACTIVE -- must not be linked.
    inert = Candidate(lemma="inert", pos="ADJ")
    inert.definition = "Sluggish; lacking vigor."
    sluggard = Candidate(lemma="sluggard", pos="NOUN")
    sluggard.definition = "A person who is inert and lazy."

    db.sync_book_results(
        conn, "Book One",
        kept=[abstruse, recondite, vilify, banisher, amble, inert, sluggard],
        rejected=[], schema=schema,
    )
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {schema}.word SET active=false WHERE lemma='inert'")
    conn.commit()

    stats = db.compute_definition_links(conn, schema)
    assert stats["words_examined"] == 6   # 7 inserted, "inert" excluded (inactive)
    assert stats["words_with_links"] == 2  # recondite, banisher
    assert stats["links_created"] == 2

    def _links_for(lemma):
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT w2.lemma, wdl.surface FROM {schema}.word_definition_link wdl
                    JOIN {schema}.word w1 ON w1.id = wdl.source_word_id
                    JOIN {schema}.word w2 ON w2.id = wdl.target_word_id
                    WHERE w1.lemma = %s""",
                (lemma,),
            )
            return cur.fetchall()

    assert _links_for("recondite") == [("abstruse", "Abstruse")]
    assert _links_for("banisher") == [("vilify", "vilified")]
    assert _links_for("amble") == []       # no self-link
    assert _links_for("sluggard") == []    # target ("inert") is inactive

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_definition_links_is_idempotent_after_definition_change():
    from concordance.model import Candidate

    schema = "cc_test_definition_links_idempotent"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    verbose = Candidate(lemma="verbose", pos="ADJ")
    verbose.definition = "Using more words than needed."
    prolix = Candidate(lemma="prolix", pos="ADJ")
    prolix.definition = "Tediously verbose."
    db.sync_book_results(conn, "Book One", kept=[verbose, prolix], rejected=[], schema=schema)

    stats = db.compute_definition_links(conn, schema)
    assert stats["links_created"] == 1
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {schema}.word_definition_link")
        assert cur.fetchone() == (1,)

    # Definition rewritten to no longer mention "verbose" -- rerunning must
    # remove the now-stale link, not just leave it stranded.
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {schema}.word SET definition='Excessively long-winded.' WHERE lemma='prolix'")
    conn.commit()

    stats2 = db.compute_definition_links(conn, schema)
    assert stats2["links_created"] == 0
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {schema}.word_definition_link")
        assert cur.fetchone() == (0,)

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_book_similarity_idf_weighting_and_thresholds():
    # 4 books. book1/book2 share 4 rare words (each appearing in ONLY those
    # two books) -- should score highly and be stored both directions.
    # book1/book4 share exactly 1 rare word -- below min_shared_words,
    # must NOT be stored. A word common to ALL 4 books must be excluded
    # from scoring entirely (max_df_fraction) and not inflate shared_word_count.
    # book3 shares nothing with anyone.
    from concordance.model import Candidate

    schema = "cc_test_book_similarity"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    def word(lemma):
        c = Candidate(lemma=lemma, pos="NOUN")
        c.definition = f"definition of {lemma}"
        return c

    common = word("commonword")
    rares = [word(f"rareword{i}") for i in range(4)]
    onerare = word("onerare")
    unique3 = [word(f"uniqueword{i}") for i in range(3)]

    db.sync_book_results(conn, "Book One", kept=[common, *rares, onerare], rejected=[], schema=schema)
    db.sync_book_results(conn, "Book Two", kept=[common, *rares], rejected=[], schema=schema)
    db.sync_book_results(conn, "Book Three", kept=[common, *unique3], rejected=[], schema=schema)
    db.sync_book_results(conn, "Book Four", kept=[common, onerare], rejected=[], schema=schema)

    stats = db.compute_book_similarity(conn, schema, min_shared_words=3)
    assert stats["books"] == 4

    with conn.cursor() as cur:
        cur.execute(f"select id, title from {schema}.book")
        ids = {title: bid for bid, title in cur.fetchall()}

        # book1/book2: 4 shared rare words, both directions stored.
        cur.execute(f"""select score, shared_word_count from {schema}.book_similarity
                        where book_a_id=%s and book_b_id=%s""", (ids["Book One"], ids["Book Two"]))
        row = cur.fetchone()
        assert row is not None
        score, shared_count = row
        assert shared_count == 4          # the common word must NOT be counted
        # book1 has one extra idf-included word (onerare) book2 doesn't share,
        # so cosine is 4/(sqrt(5)*sqrt(4)) = 2/sqrt(5) ~= 0.894, not 1.0.
        assert 0.85 < score < 0.95
        cur.execute(f"""select score, shared_word_count from {schema}.book_similarity
                        where book_a_id=%s and book_b_id=%s""", (ids["Book Two"], ids["Book One"]))
        assert cur.fetchone() == (score, shared_count)   # symmetric, both directions stored

        # book1/book4: only 1 shared rare word -- below min_shared_words, not stored.
        cur.execute(f"""select 1 from {schema}.book_similarity
                        where book_a_id=%s and book_b_id=%s""", (ids["Book One"], ids["Book Four"]))
        assert cur.fetchone() is None

        # book3 shares nothing -- no rows at all involving it.
        cur.execute(f"""select count(*) from {schema}.book_similarity
                        where book_a_id=%s or book_b_id=%s""", (ids["Book Three"], ids["Book Three"]))
        assert cur.fetchone() == (0,)

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_book_similarity_respects_top_k_and_is_idempotent():
    from concordance.model import Candidate

    schema = "cc_test_book_similarity_topk"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    def word(lemma):
        c = Candidate(lemma=lemma, pos="NOUN")
        c.definition = f"definition of {lemma}"
        return c

    # "hub" shares 3 rare words with each of 3 other books -- with top_k=2,
    # only its 2 best-scoring neighbors should be stored.
    shared_sets = [[word(f"book{b}word{i}") for i in range(3)] for b in range(3)]
    hub_words = [w for group in shared_sets for w in group]
    db.sync_book_results(conn, "Hub", kept=hub_words, rejected=[], schema=schema)
    for b in range(3):
        db.sync_book_results(conn, f"Leaf{b}", kept=shared_sets[b], rejected=[], schema=schema)

    stats1 = db.compute_book_similarity(conn, schema, top_k=2, min_shared_words=3)
    assert stats1["books"] == 4

    with conn.cursor() as cur:
        cur.execute(f"select id from {schema}.book where title='Hub'")
        hub_id = cur.fetchone()[0]
        cur.execute(f"select count(*) from {schema}.book_similarity where book_a_id=%s", (hub_id,))
        assert cur.fetchone() == (2,)   # capped at top_k, not all 3 leaves

    # Re-running must not duplicate rows (always-recompute, not append).
    db.compute_book_similarity(conn, schema, top_k=2, min_shared_words=3)
    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {schema}.book_similarity where book_a_id=%s", (hub_id,))
        assert cur.fetchone() == (2,)

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_author_similarity_idf_weighting_and_thresholds():
    # Same shape as test_compute_book_similarity_idf_weighting_and_thresholds,
    # one book per author -- author-df here is identical to book-df since
    # each author only has one book, so the expected numbers match exactly.
    # The author-vs-book-df distinction (an author with SEVERAL books
    # containing the same word must count once, not once per book) is
    # covered separately below via the DISTINCT-authors_by_word case.
    from concordance.model import Candidate

    schema = "cc_test_author_similarity"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    def word(lemma):
        c = Candidate(lemma=lemma, pos="NOUN")
        c.definition = f"definition of {lemma}"
        return c

    common = word("commonword")
    rares = [word(f"rareword{i}") for i in range(4)]
    onerare = word("onerare")
    unique3 = [word(f"uniqueword{i}") for i in range(3)]

    db.sync_book_results(conn, "Book One", kept=[common, *rares, onerare], rejected=[], schema=schema, author="Author One")
    db.sync_book_results(conn, "Book Two", kept=[common, *rares], rejected=[], schema=schema, author="Author Two")
    db.sync_book_results(conn, "Book Three", kept=[common, *unique3], rejected=[], schema=schema, author="Author Three")
    db.sync_book_results(conn, "Book Four", kept=[common, onerare], rejected=[], schema=schema, author="Author Four")

    stats = db.compute_author_similarity(conn, schema, min_shared_words=3)
    assert stats["authors"] == 4

    with conn.cursor() as cur:
        cur.execute(f"""select score, shared_word_count from {schema}.author_similarity
                        where author_a=%s and author_b=%s""", ("Author One", "Author Two"))
        row = cur.fetchone()
        assert row is not None
        score, shared_count = row
        assert shared_count == 4          # the common word must NOT be counted
        assert 0.85 < score < 0.95        # same math as the book-level test: 2/sqrt(5)

        cur.execute(f"""select score, shared_word_count from {schema}.author_similarity
                        where author_a=%s and author_b=%s""", ("Author Two", "Author One"))
        assert cur.fetchone() == (score, shared_count)   # symmetric, both directions stored

        # Author One/Four: only 1 shared rare word -- below min_shared_words.
        cur.execute(f"""select 1 from {schema}.author_similarity
                        where author_a=%s and author_b=%s""", ("Author One", "Author Four"))
        assert cur.fetchone() is None

        # Author Three shares nothing -- no rows at all involving it.
        cur.execute(f"""select count(*) from {schema}.author_similarity
                        where author_a=%s or author_b=%s""", ("Author Three", "Author Three"))
        assert cur.fetchone() == (0,)

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_author_similarity_dedupes_shared_words_across_an_authors_own_books():
    # The one case that genuinely diverges from book-level math: an author
    # with SEVERAL books containing the same word must have that word count
    # once toward their own vector, not once per book -- otherwise the
    # DISTINCT in compute_author_similarity's authors_by_word query would be
    # a no-op and this test would silently pass even if it were removed.
    from concordance.model import Candidate

    schema = "cc_test_author_similarity_dedupe"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    def word(lemma):
        c = Candidate(lemma=lemma, pos="NOUN")
        c.definition = f"definition of {lemma}"
        return c

    shared = [word(f"shared{i}") for i in range(3)]
    # Two more, unrelated authors: with only {Prolific, Solo} (n_authors=2),
    # max_df_fraction (0.5 * 2 = 1) would exclude `shared` (author-df=2)
    # entirely -- same trap the book-level tests route around with a 4th
    # book. Two fillers bring n_authors to 4, max_df=2, so df=2 survives.
    db.sync_book_results(conn, "Prolific Book A", kept=list(shared), rejected=[], schema=schema, author="Prolific")
    db.sync_book_results(conn, "Prolific Book B", kept=list(shared), rejected=[], schema=schema, author="Prolific")
    db.sync_book_results(conn, "Solo Book", kept=list(shared), rejected=[], schema=schema, author="Solo")
    db.sync_book_results(conn, "Filler Book", kept=[word("fillerword")], rejected=[], schema=schema, author="Filler")
    db.sync_book_results(conn, "Filler2 Book", kept=[word("filler2word")], rejected=[], schema=schema, author="Filler2")

    db.compute_author_similarity(conn, schema, min_shared_words=3)

    with conn.cursor() as cur:
        cur.execute(f"""select shared_word_count from {schema}.author_similarity
                        where author_a=%s and author_b=%s""", ("Prolific", "Solo"))
        row = cur.fetchone()
        assert row is not None
        assert row[0] == 3   # not 6 -- each shared word counts once per author, not once per book

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_author_clustering_separates_two_clear_clusters():
    # 10 authors, two disjoint 10-word vocabularies (5 authors each) -- the
    # clearest possible synthetic multi-cluster fixture: any reasonable
    # clustering must put all 5 "alpha" authors in one cluster and all 5
    # "beta" authors in another, and MDS must place them on opposite sides
    # of the map (opposite-signed x-coordinates), not just "different
    # clusters" -- the two views (color, position) must agree.
    from concordance.model import Candidate

    schema = "cc_test_author_clustering"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    def word(lemma):
        c = Candidate(lemma=lemma, pos="NOUN")
        c.definition = f"definition of {lemma}"
        return c

    alpha_words = [word(f"alphaword{i}") for i in range(10)]
    beta_words = [word(f"betaword{i}") for i in range(10)]
    for i in range(5):
        db.sync_book_results(conn, f"Alpha Book {i}", kept=alpha_words, rejected=[], schema=schema, author=f"Alpha{i}")
    for i in range(5):
        db.sync_book_results(conn, f"Beta Book {i}", kept=beta_words, rejected=[], schema=schema, author=f"Beta{i}")

    stats = db.compute_author_clustering(conn, schema, top_n=200, n_clusters=2)
    assert stats["authors"] == 10
    assert stats["clusters"] == 2

    with conn.cursor() as cur:
        cur.execute(f"select author, cluster_id, mds_x from {schema}.author_cluster order by author")
        rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

        alpha_clusters = {rows[f"Alpha{i}"][0] for i in range(5)}
        beta_clusters = {rows[f"Beta{i}"][0] for i in range(5)}
        assert len(alpha_clusters) == 1          # all 5 alphas in the same cluster
        assert len(beta_clusters) == 1            # all 5 betas in the same cluster
        assert alpha_clusters != beta_clusters    # and it's a DIFFERENT cluster from the alphas

        alpha_x = [rows[f"Alpha{i}"][1] for i in range(5)]
        beta_x = [rows[f"Beta{i}"][1] for i in range(5)]
        # Same sign within each group, opposite sign between groups -- MDS
        # actually separated them spatially, not just by cluster label.
        assert all((x > 0) == (alpha_x[0] > 0) for x in alpha_x)
        assert all((x > 0) == (beta_x[0] > 0) for x in beta_x)
        assert (alpha_x[0] > 0) != (beta_x[0] > 0)

        cur.execute(f"select leaf_order from {schema}.author_cluster_run where id=1")
        leaf_order = cur.fetchone()[0]
        assert set(leaf_order) == set(rows.keys())
        # Seriation groups same-cluster authors together, not interleaved --
        # every Alpha should be contiguous in leaf_order, likewise Beta.
        alpha_positions = sorted(leaf_order.index(f"Alpha{i}") for i in range(5))
        assert alpha_positions == list(range(alpha_positions[0], alpha_positions[0] + 5))

        cur.execute(f"select grid, tree_json from {schema}.author_cluster_run where id=1")
        grid, tree = cur.fetchone()
        assert len(grid) == 10 and len(grid[0]) == 10
        assert "distance" in tree and "size" in tree

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_author_clustering_min_fame_selects_by_fame_and_writes_separate_tables():
    """min_fame is a second, independent selection mode (see the function's
    own docstring): instead of top_n by book count, every author with
    author_fame.fame_score >= min_fame qualifies, written to
    author_cluster_fame/author_cluster_fame_run -- author_cluster/
    author_cluster_run (the default top_n-by-volume run) must be
    untouched, since the two views are meant to coexist."""
    from concordance.model import Candidate

    schema = "cc_test_author_clustering_fame"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    def word(lemma):
        c = Candidate(lemma=lemma, pos="NOUN")
        c.definition = f"definition of {lemma}"
        return c

    alpha_words = [word(f"alphaword{i}") for i in range(10)]
    beta_words = [word(f"betaword{i}") for i in range(10)]
    # 5 famous ("Alpha") authors, 5 obscure ("Beta") -- min_fame=8 must
    # select only the Alphas, regardless of book count (both groups have
    # one book each, so volume alone can't distinguish them).
    for i in range(5):
        db.sync_book_results(conn, f"Alpha Book {i}", kept=alpha_words, rejected=[], schema=schema, author=f"Alpha{i}")
    for i in range(5):
        db.sync_book_results(conn, f"Beta Book {i}", kept=beta_words, rejected=[], schema=schema, author=f"Beta{i}")
    with conn.cursor() as cur:
        for i in range(5):
            cur.execute(f"INSERT INTO {schema}.author_fame (author, fame_score) VALUES (%s, 9.0)", (f"Alpha{i}",))
        for i in range(5):
            cur.execute(f"INSERT INTO {schema}.author_fame (author, fame_score) VALUES (%s, 3.0)", (f"Beta{i}",))
    conn.commit()

    # A prior top_n-by-volume run -- must survive the min_fame run untouched.
    db.compute_author_clustering(conn, schema, top_n=200, n_clusters=2)
    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {schema}.author_cluster")
        volume_count_before = cur.fetchone()[0]

    stats = db.compute_author_clustering(conn, schema, min_fame=8.0, n_clusters=2)
    assert stats["authors"] == 5

    with conn.cursor() as cur:
        cur.execute(f"select author from {schema}.author_cluster_fame order by author")
        fame_authors = {r[0] for r in cur.fetchall()}
        assert fame_authors == {f"Alpha{i}" for i in range(5)}

        cur.execute(f"select count(*) from {schema}.author_cluster_fame_run")
        assert cur.fetchone()[0] == 1

        cur.execute(f"select count(*) from {schema}.author_cluster")
        assert cur.fetchone()[0] == volume_count_before == 10

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_book_clustering_min_fame_selects_by_fame_and_writes_separate_tables():
    """Book-level counterpart to the author min_fame test above -- same
    selection swap (book_fame.fame_score >= min_fame instead of top_n by
    word count), same separate destination tables (book_cluster_fame/
    book_cluster_fame_run), same untouched-original-run guarantee."""
    from concordance.model import Candidate

    schema = "cc_test_book_clustering_fame"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    def word(lemma):
        c = Candidate(lemma=lemma, pos="NOUN")
        c.definition = f"definition of {lemma}"
        return c

    alpha_words = [word(f"alphaword{i}") for i in range(10)]
    beta_words = [word(f"betaword{i}") for i in range(10)]
    alpha_ids, beta_ids = [], []
    for i in range(5):
        db.sync_book_results(conn, f"Alpha Book {i}", kept=alpha_words, rejected=[], schema=schema, author="Alpha")
    for i in range(5):
        db.sync_book_results(conn, f"Beta Book {i}", kept=beta_words, rejected=[], schema=schema, author="Beta")
    with conn.cursor() as cur:
        cur.execute(f"select id, title from {schema}.book")
        for bid, title in cur.fetchall():
            (alpha_ids if title.startswith("Alpha") else beta_ids).append(bid)
        for bid in alpha_ids:
            cur.execute(f"INSERT INTO {schema}.book_fame (book_id, fame_score) VALUES (%s, 9.0)", (bid,))
        for bid in beta_ids:
            cur.execute(f"INSERT INTO {schema}.book_fame (book_id, fame_score) VALUES (%s, 3.0)", (bid,))
    conn.commit()

    db.compute_book_clustering(conn, schema, top_n=200, n_clusters=2)
    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {schema}.book_cluster")
        volume_count_before = cur.fetchone()[0]

    stats = db.compute_book_clustering(conn, schema, min_fame=8.0, n_clusters=2)
    assert stats["books"] == 5

    with conn.cursor() as cur:
        cur.execute(f"select book_id from {schema}.book_cluster_fame")
        assert {r[0] for r in cur.fetchall()} == set(alpha_ids)

        cur.execute(f"select count(*) from {schema}.book_cluster_fame_run")
        assert cur.fetchone()[0] == 1

        cur.execute(f"select count(*) from {schema}.book_cluster")
        assert cur.fetchone()[0] == volume_count_before == 10

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_author_clustering_is_idempotent_and_deterministic():
    from concordance.model import Candidate

    schema = "cc_test_author_clustering_idempotent"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    def word(lemma):
        c = Candidate(lemma=lemma, pos="NOUN")
        c.definition = f"definition of {lemma}"
        return c

    alpha_words = [word(f"alphaword{i}") for i in range(10)]
    beta_words = [word(f"betaword{i}") for i in range(10)]
    for i in range(5):
        db.sync_book_results(conn, f"Alpha Book {i}", kept=alpha_words, rejected=[], schema=schema, author=f"Alpha{i}")
    for i in range(5):
        db.sync_book_results(conn, f"Beta Book {i}", kept=beta_words, rejected=[], schema=schema, author=f"Beta{i}")

    db.compute_author_clustering(conn, schema, top_n=200, n_clusters=2)
    with conn.cursor() as cur:
        cur.execute(f"select author, cluster_id, mds_x, mds_y from {schema}.author_cluster order by author")
        run1 = cur.fetchall()
        cur.execute(f"select count(*) from {schema}.author_cluster")
        count1 = cur.fetchone()[0]

    db.compute_author_clustering(conn, schema, top_n=200, n_clusters=2)
    with conn.cursor() as cur:
        cur.execute(f"select author, cluster_id, mds_x, mds_y from {schema}.author_cluster order by author")
        run2 = cur.fetchall()
        cur.execute(f"select count(*) from {schema}.author_cluster")
        count2 = cur.fetchone()[0]

        # Re-running must truncate + repopulate, not append -- same row
        # count both times, not double.
        assert count1 == count2 == 10
        # Eigenvector sign is pinned deterministically, so a re-run on
        # unchanged data must reproduce bit-identical coordinates, not
        # mirror-flip.
        assert run1 == run2

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


# --- compute_personal_difficulty (concordance/calibration.py) -------------

def _cal_user(cur, schema, username):
    cur.execute(f"INSERT INTO {schema}.users (username, password_hash) VALUES (%s, 'x') RETURNING id",
                (username,))
    return cur.fetchone()[0]


def _cal_word(cur, schema, lemma, difficulty):
    cur.execute(f"INSERT INTO {schema}.word (lemma, definition) VALUES (%s, %s) RETURNING id",
                (lemma, f"definition of {lemma}"))
    wid = cur.fetchone()[0]
    cur.execute(f"INSERT INTO {schema}.word_difficulty (word_id, difficulty) VALUES (%s, %s)",
                (wid, difficulty))
    return wid


def _cal_answer(cur, schema, user_id, word_id, is_correct, guessing_floor, answered_at):
    """One quiz_answer row, each wrapped in its own session/question (a real
    session would batch several answers into one, but compute_personal_difficulty
    only cares about answered_at ordering per (user, word), not session shape)."""
    cur.execute(f"""INSERT INTO {schema}.quiz_session (user_id, config, feedback_timing)
                    VALUES (%s, '{{}}', 'immediate') RETURNING id""", (user_id,))
    session_id = cur.fetchone()[0]
    cur.execute(f"""INSERT INTO {schema}.quiz_question
                        (session_id, seq, question_type, target_word_ids, payload)
                    VALUES (%s, 1, 'mc', %s, '{{}}') RETURNING id""", (session_id, [word_id]))
    question_id = cur.fetchone()[0]
    cur.execute(f"""INSERT INTO {schema}.quiz_answer
                        (question_id, word_id, response, is_correct, guessing_floor,
                         question_type, direction, answered_at)
                    VALUES (%s, %s, '{{}}', %s, %s, 'mc', 'definition_to_word', %s)""",
                (question_id, word_id, is_correct, guessing_floor, answered_at))


@pg
def test_compute_personal_difficulty_never_quizzed_word_gets_no_row():
    schema = "cc_test_calibration_unquizzed"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    with conn.cursor() as cur:
        _cal_user(cur, schema, "calibuser")
        _cal_word(cur, schema, "untouchedword", 50.0)
    conn.commit()

    stats = db.compute_personal_difficulty(conn, schema)
    assert stats["words"] == 0

    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {schema}.word_personal_difficulty")
        assert cur.fetchone() == (0,)
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_personal_difficulty_correct_and_incorrect_move_opposite_ways():
    schema = "cc_test_calibration_updown"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    with conn.cursor() as cur:
        user_id = _cal_user(cur, schema, "calibuser")
        easy_correct = _cal_word(cur, schema, "easycorrectword", 50.0)
        hard_wrong = _cal_word(cur, schema, "hardwrongword", 50.0)
        # 4-option mc -> guessing_floor 0.25.
        _cal_answer(cur, schema, user_id, easy_correct, True, 0.25, "2026-01-01T00:00:00Z")
        _cal_answer(cur, schema, user_id, hard_wrong, False, 0.25, "2026-01-01T00:00:00Z")
    conn.commit()

    stats = db.compute_personal_difficulty(conn, schema)
    assert stats["words"] == 2

    with conn.cursor() as cur:
        cur.execute(f"""select word_id, personal_difficulty, based_on_correct
                        from {schema}.word_personal_difficulty order by word_id""")
        rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
        correct_diff, correct_flag = rows[easy_correct]
        wrong_diff, wrong_flag = rows[hard_wrong]

        assert correct_flag is True
        assert wrong_flag is False
        # A correct first answer should make the word look EASIER than the
        # ex-ante 50, an incorrect one HARDER -- and the move must be a
        # bounded nudge (calibration.py's DEFAULT_ETA=1.0 nudge), not a
        # wild swing to 0/100.
        assert 0 < correct_diff < 50
        assert 50 < wrong_diff < 100

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_personal_difficulty_only_first_exposure_counts():
    # Same (user, word) answered twice: an early WRONG answer followed by a
    # later CORRECT one must calibrate off the early wrong answer only --
    # the later "he learned it" re-exposure must not change the stored
    # value (see compute_personal_difficulty's own docstring on why).
    schema = "cc_test_calibration_first_exposure"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    with conn.cursor() as cur:
        user_id = _cal_user(cur, schema, "calibuser")
        word_id = _cal_word(cur, schema, "relearnedword", 50.0)
        _cal_answer(cur, schema, user_id, word_id, False, 0.25, "2026-01-01T00:00:00Z")
        _cal_answer(cur, schema, user_id, word_id, True, 0.25, "2026-06-01T00:00:00Z")
    conn.commit()

    db.compute_personal_difficulty(conn, schema)
    with conn.cursor() as cur:
        cur.execute(f"""select personal_difficulty, based_on_correct
                        from {schema}.word_personal_difficulty where word_id=%s""", (word_id,))
        diff, based_on_correct = cur.fetchone()
        assert based_on_correct is False   # the FIRST (wrong) answer, not the later correct one
        assert diff > 50                   # moved harder, not easier

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_personal_difficulty_skips_words_without_ex_ante_baseline():
    schema = "cc_test_calibration_no_baseline"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    with conn.cursor() as cur:
        user_id = _cal_user(cur, schema, "calibuser")
        cur.execute(f"INSERT INTO {schema}.word (lemma, definition) VALUES ('nobaselineword', 'def') RETURNING id")
        word_id = cur.fetchone()[0]
        # No word_difficulty row at all -- nothing to anchor a nudge to.
        _cal_answer(cur, schema, user_id, word_id, True, 0.25, "2026-01-01T00:00:00Z")
    conn.commit()

    stats = db.compute_personal_difficulty(conn, schema)
    assert stats["words"] == 0
    assert stats["skipped_no_baseline"] == 1

    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {schema}.word_personal_difficulty")
        assert cur.fetchone() == (0,)
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_personal_difficulty_is_idempotent():
    schema = "cc_test_calibration_idempotent"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    with conn.cursor() as cur:
        user_id = _cal_user(cur, schema, "calibuser")
        word_id = _cal_word(cur, schema, "idempotentword", 50.0)
        _cal_answer(cur, schema, user_id, word_id, True, 0.25, "2026-01-01T00:00:00Z")
    conn.commit()

    db.compute_personal_difficulty(conn, schema)
    with conn.cursor() as cur:
        cur.execute(f"select item_rating, personal_difficulty from {schema}.word_personal_difficulty")
        run1 = cur.fetchall()

    db.compute_personal_difficulty(conn, schema)
    with conn.cursor() as cur:
        cur.execute(f"select item_rating, personal_difficulty from {schema}.word_personal_difficulty")
        run2 = cur.fetchall()

        assert run1 == run2   # unchanged response data -> bit-identical recompute

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()
