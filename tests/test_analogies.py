"""Analogy relation extraction + verification (§ analogies). Pure tests run
always; DB-backed tests run only when a throwaway Postgres is provided via
CONCORDANCE_TEST_DB_URL (else skipped) -- same convention as test_distractors.py."""

from __future__ import annotations

import os

import pytest

from concordance import analogies as an
from concordance import analogy_select as asel
from concordance import db
from concordance.model import RelationCandidate, wordnet_pos


# --- pure helpers (no database, no LLM) ---------------------------------------

def test_wordnet_pos_mapping():
    assert wordnet_pos("noun") == "n"
    assert wordnet_pos("verb") == "v"
    assert wordnet_pos("adjective") == "a"
    assert wordnet_pos("adverb") == "r"
    assert wordnet_pos("Noun") == "n"          # normalize_pos folds case
    assert wordnet_pos("preposition") is None  # no WordNet code for this POS
    assert wordnet_pos(None) is None


def test_relation_family_covers_every_relation_type_used_in_schema():
    # concordance/db.py's word_relation_edge.relation_type comment enumerates
    # these exact values -- every one must have a family bucket, or an edge of
    # that type could never be paired with an anchor (see analogy_select.py).
    expected_types = {
        "hypernym", "holonym_part", "holonym_member", "holonym_substance",
        "antonym", "similar_to", "derivationally_related", "attribute",
        "definition_pattern_kind_of", "definition_pattern_agent",
        "definition_pattern_part_of", "definition_pattern_purpose",
        "definition_pattern_relates_to", "definition_pattern_resembling",
        "definition_pattern_characterized_by",
    }
    assert expected_types == set(an._RELATION_FAMILY)


def _mk(a="cangue", b="collar"):
    return RelationCandidate(a, "noun", f"{a} def", b, "noun", f"{b} def", "hypernym", "is_a", "wordnet_hypernym")


def test_apply_verify_verdicts_reject_biased_on_omission():
    """The single most safety-critical test in this feature: an omitted or
    unparseable verdict must default to REJECTED, the opposite of judge.py's
    keep-bias -- an unverified pair shipping means a live quiz question with
    two right answers, so silence must mean discard."""
    batch = [_mk("cangue", "collar"), _mk("fetter", "shackle"), _mk("plotter", "contriver")]
    an.apply_verify_verdicts(batch, [{"i": 0, "ok": True}, {"i": 1, "ok": True}])  # index 2 omitted
    assert [c.verdict for c in batch] == ["verified", "verified", "rejected"]


def test_apply_verify_verdicts_reject_biased_on_unparseable_response():
    batch = [_mk("a", "b"), _mk("c", "d")]
    an.apply_verify_verdicts(batch, None)
    assert all(c.verdict == "rejected" for c in batch)


def test_apply_verify_verdicts_explicit_false_and_true():
    batch = [_mk("x", "y")]
    an.apply_verify_verdicts(batch, [{"i": 0, "ok": False}])
    assert batch[0].verdict == "rejected"

    batch2 = [_mk("x", "y")]
    an.apply_verify_verdicts(batch2, [{"i": 0, "ok": True}])
    assert batch2[0].verdict == "verified"


def test_definition_pattern_matchers():
    nlp = an._get_nlp()
    matcher = an._build_matchers(nlp)

    def labels_for(text):
        doc = nlp(text)
        return sorted({nlp.vocab.strings[m[0]] for m in matcher(doc)})

    assert labels_for("a kind of large wild cat") == ["kind_of"]
    assert labels_for("a type of ancient Greek pottery") == ["kind_of"]
    assert labels_for("one who steals from a grave") == ["agent"]
    assert labels_for("one who habitually tells lies") == ["agent"]
    assert labels_for("a small part of the engine") == ["part_of"]
    assert labels_for("used for cutting wood") == ["purpose"]
    # relates_to/resembling/characterized_by: added specifically for
    # adjectives, where WordNet's own relation graph is almost empty (see
    # _build_matchers' docstring).
    assert labels_for("of or relating to business") == ["relates_to"]
    assert labels_for("relating to the human heart") == ["relates_to"]
    assert labels_for("resembling a small bear") == ["resembling"]
    assert labels_for("like a sword") == ["resembling"]
    assert labels_for("having a bushy tail") == ["characterized_by"]
    assert labels_for("characterized by rapid growth") == ["characterized_by"]
    # A bare-NP definition with no signal phrase at all -- none of the seven
    # patterns should fire (this is exactly the class of definition the
    # patterns are NOT meant to cover; see the module docstring).
    assert labels_for("a heavy wooden collar worn by prisoners") == []


# --- DB-backed -----------------------------------------------------------------

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


def _seed_word(cur, schema, lemma, definition, pos="noun"):
    cur.execute(
        f"""INSERT INTO {schema}.word (lemma, definition, part_of_speech, active)
            VALUES (%s, %s, %s, true) RETURNING id""",
        (lemma, definition, pos),
    )
    wid = cur.fetchone()[0]
    cur.execute(f"INSERT INTO {schema}.word_difficulty (word_id, quizzable) VALUES (%s, true)", (wid,))
    return wid


@pg
def test_extract_wordnet_edges_and_definition_patterns():
    schema = "cc_test_analogies_extract"
    conn = db.connect(_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    db.apply_schema(conn, schema)
    try:
        with conn.cursor() as cur:
            wid_fetter = _seed_word(cur, schema, "fetter", "a chain or shackle placed around the ankles")
            wid_malefactor = _seed_word(cur, schema, "malefactor", "one who commits a crime or offense")

        wn = an._load_wordnet()
        assert wn is not None
        terms = an.resolve_vocab_terms(conn, schema, wn)
        by_lemma = {t["lemma"]: t for t in terms}
        assert by_lemma["fetter"]["synset_name"] is not None

        nlp = an._get_nlp()
        wn_edges, fanout_n = an.extract_wordnet_edges(conn, schema, wn, by_lemma["fetter"])
        assert any(c.relation_type == "hypernym" for c in wn_edges)
        assert all(c.term_a_gloss and c.term_b_gloss for c in wn_edges), \
            "every candidate must carry real gloss text -- it's what the verification gate judges"

        def_edges = an.extract_definition_pattern_edges(conn, schema, wn, nlp, by_lemma["malefactor"])
        assert any(c.relation_type == "definition_pattern_agent" and c.term_b_lemma == "commit"
                   for c in def_edges)
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


@pg
def test_no_duplicate_term_row_when_lemma_is_both_vocab_and_anchor():
    """_get_or_create_ordinary_term must never create a second word_id=NULL row
    for a lemma that's already a vocab word (see _find_term_id's docstring) --
    otherwise analogy_select.py could resolve the same lemma to two different
    term ids depending on lookup path."""
    schema = "cc_test_analogies_no_dup"
    conn = db.connect(_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    db.apply_schema(conn, schema)
    try:
        with conn.cursor() as cur:
            wid = _seed_word(cur, schema, "shackle", "a metal restraint for the wrists or ankles")
        wn = an._load_wordnet()
        an.resolve_vocab_terms(conn, schema, wn)
        with conn.cursor() as cur:
            other = an._get_or_create_ordinary_term(cur, schema, wn, "shackle", "n")
            assert other is not None
            assert other["word_id"] == wid
            cur.execute(f"SELECT count(*) FROM {schema}.wn_relation_term WHERE lemma_lc = 'shackle'")
            assert cur.fetchone()[0] == 1
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


@pg
def test_select_analogy_edge_two_hop_transitive_closure_exclusion():
    """Regression test: the exclusion set for a tested hypernym edge must
    include the FULL closure (grandparent and beyond), not just the direct
    target -- getting this wrong ships items with a second valid answer."""
    schema = "cc_test_analogies_closure"
    conn = db.connect(_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    db.apply_schema(conn, schema)
    try:
        with conn.cursor() as cur:
            wid = {}
            for lemma in ("fetter", "shackle", "gauntlet", "vambrace"):
                wid[lemma] = _seed_word(cur, schema, lemma, f"{lemma} definition")

            def add_term(lemma, word_id=None):
                cur.execute(
                    f"""INSERT INTO {schema}.wn_relation_term (word_id, lemma, wn_pos, synset_name, gloss, is_common)
                        VALUES (%s, %s, 'n', %s, %s, %s) RETURNING id""",
                    (word_id, lemma, f"{lemma}.n.01", f"{lemma} gloss", word_id is None),
                )
                return cur.fetchone()[0]

            t_fetter = add_term("fetter", wid["fetter"])
            t_shackle = add_term("shackle", wid["shackle"])
            t_gauntlet = add_term("gauntlet", wid["gauntlet"])
            t_vambrace = add_term("vambrace", wid["vambrace"])

            cur.execute(
                f"""INSERT INTO {schema}.word_relation_edge
                        (term_a_id, term_b_id, relation_type, relation_family, pos_a, pos_b, source, verification_status)
                    VALUES (%s,%s,'hypernym','is_a','noun','noun','wordnet_hypernym','verified')""",
                (t_fetter, t_shackle),
            )
            cur.execute(
                f"""INSERT INTO {schema}.word_relation_edge
                        (term_a_id, term_b_id, relation_type, relation_family, pos_a, pos_b, source, verification_status)
                    VALUES (%s,%s,'hypernym','is_a','noun','noun','wordnet_hypernym','verified')""",
                (t_gauntlet, t_vambrace),
            )
            # simulated two-hop closure: fetter's hypernym fanout includes both
            # the direct target (shackle) and a grandparent (restraint), which a
            # buggy direct-only exclusion set would miss.
            for lemma in ("shackle", "restraint", "trammel"):
                cur.execute(
                    f"""INSERT INTO {schema}.wn_relation_fanout (term_id, relation_type, target_lemma, target_pos)
                        VALUES (%s, 'hypernym', %s, 'n')""",
                    (t_fetter, lemma),
                )

        assembly = asel.select_analogy_edge(conn, schema, wid["fetter"], exclude_ids=set())
        assert assembly is not None
        assert assembly.c_lemma == "fetter" and assembly.d_lemma == "shackle"
        assert {"shackle", "restraint", "trammel"} <= assembly.exclusion_lemmas
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def test_too_generic_for_trap_matches_entitys_own_direct_hyponyms():
    # _TOO_GENERIC_FOR_TRAP should be exactly {"entity"} union entity.n.01's
    # own direct hyponym lemmas (confirmed against nltk's wordnet, not
    # hand-picked) -- every non-"entity" member must actually be a direct
    # hyponym, not merely overlap with one; this would have caught leaving
    # a real hyponym (like "thing") off the list without erroring.
    try:
        from nltk.corpus import wordnet as wn
        hyponym_lemmas = {l.replace("_", " ") for h in wn.synset("entity.n.01").hyponyms()
                          for l in h.lemma_names()}
    except LookupError:
        pytest.skip("nltk wordnet corpus not downloaded")
    assert asel._TOO_GENERIC_FOR_TRAP - {"entity"} <= hyponym_lemmas


@pg
def test_select_analogy_edge_strips_wordnet_root_lemmas_from_trap_pool():
    # Regression test for a live-reported bug: "abstraction"/"abstract
    # entity" showing up as an analogy distractor on nearly every is_a
    # question, because entity.n.01 (WordNet's sole noun-hierarchy root)
    # and its direct hyponyms sit atop virtually every noun's transitive
    # hypernym closure -- confirmed live against the real corpus: 'entity'
    # appears in 80% of terms' hypernym fanout, 'abstraction'/'abstract
    # entity' in 45%. trap_lemmas must exclude them (too generic to be a
    # meaningful "plausible wrong answer") while still keeping a real,
    # specific trap candidate from the same fanout.
    schema = "cc_test_analogies_generic_trap"
    conn = db.connect(_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    db.apply_schema(conn, schema)
    try:
        with conn.cursor() as cur:
            wid = {}
            for lemma in ("fetter", "shackle", "gauntlet", "vambrace"):
                wid[lemma] = _seed_word(cur, schema, lemma, f"{lemma} definition")

            def add_term(lemma, word_id=None):
                cur.execute(
                    f"""INSERT INTO {schema}.wn_relation_term (word_id, lemma, wn_pos, synset_name, gloss, is_common)
                        VALUES (%s, %s, 'n', %s, %s, %s) RETURNING id""",
                    (word_id, lemma, f"{lemma}.n.01", f"{lemma} gloss", word_id is None),
                )
                return cur.fetchone()[0]

            t_fetter = add_term("fetter", wid["fetter"])
            t_shackle = add_term("shackle", wid["shackle"])
            t_gauntlet = add_term("gauntlet", wid["gauntlet"])
            t_vambrace = add_term("vambrace", wid["vambrace"])

            cur.execute(
                f"""INSERT INTO {schema}.word_relation_edge
                        (term_a_id, term_b_id, relation_type, relation_family, pos_a, pos_b, source, verification_status)
                    VALUES (%s,%s,'hypernym','is_a','noun','noun','wordnet_hypernym','verified')""",
                (t_fetter, t_shackle),
            )
            cur.execute(
                f"""INSERT INTO {schema}.word_relation_edge
                        (term_a_id, term_b_id, relation_type, relation_family, pos_a, pos_b, source, verification_status)
                    VALUES (%s,%s,'hypernym','is_a','noun','noun','wordnet_hypernym','verified')""",
                (t_gauntlet, t_vambrace),
            )
            # gauntlet (the anchor's A side) has both a real, specific
            # trap candidate (armor) AND the generic root cluster in its
            # hypernym fanout -- the fix must drop only the latter.
            for lemma in ("armor", "entity", "abstraction", "abstract entity", "physical entity"):
                cur.execute(
                    f"""INSERT INTO {schema}.wn_relation_fanout (term_id, relation_type, target_lemma, target_pos)
                        VALUES (%s, 'hypernym', %s, 'n')""",
                    (t_gauntlet, lemma),
                )

        assembly = asel.select_analogy_edge(conn, schema, wid["fetter"], exclude_ids=set())
        assert assembly is not None
        assert not (assembly.trap_lemmas and set(assembly.trap_lemmas) & asel._TOO_GENERIC_FOR_TRAP)
        assert "armor" in assembly.trap_lemmas
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
