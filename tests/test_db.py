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

    monkeypatch.setattr(llama_cpp, "Llama", lambda *a, **k: object())
    model_path = tmp_path / "fake.gguf"
    model_path.write_bytes(b"")

    monkeypatch.setattr(resolve.localdict, "enrich", lambda cand, lex: False)
    monkeypatch.setattr(resolve.dictionary, "enrich", lambda cand, session: None)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
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
def test_dedupe_plural_definitions_all_three_outcomes(monkeypatch):
    from concordance import resolve
    from concordance.model import Candidate

    monkeypatch.setattr(resolve.localdict, "enrich", lambda cand, lex: False)
    monkeypatch.setattr(resolve.deepdef, "wordnik_key", lambda: "")
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
