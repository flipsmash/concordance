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
    # concordance/db's word_relation_edge.relation_type comment enumerates
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


# --- item quality rules (2026-11-20) -------------------------------------------

def test_shares_stem_flags_spelling_giveaways_only():
    for c, d in [("torpidity", "torpid"), ("dogal", "doge"), ("imperfect", "perfect"),
                 ("functioning", "function"), ("insane", "insanity")]:
        assert asel.shares_stem(c, d), (c, d)
    for c, d in [("dryad", "nymph"), ("photomicrographs", "picture"), ("monolith", "stone")]:
        assert not asel.shares_stem(c, d), (c, d)


def test_options_clash_variants_and_overlapping_names():
    from concordance import distractors as dx

    assert dx._options_clash("artifact", "artefact")
    assert dx._options_clash("academic term", "academic session")
    assert dx._options_clash("component", "component part")
    assert dx._options_clash("cue", "clue")
    assert not dx._options_clash("primate", "placental")
    assert dx._options_clash("w.m.d.", "weapon of mass destruction")
    assert dx._options_clash("wmd", "weapon of mass destruction")
    assert dx._is_abbreviation("w.m.d.") and dx._is_abbreviation("u.s.")
    assert not dx._is_abbreviation("weapon") and not dx._is_abbreviation("st. john")


def test_taxon_answers_detected_by_gloss_or_synonym():
    assert asel._is_taxon("New World lizards", ["family iguania", "iguanidae"])
    assert asel._is_taxon("genus of deciduous trees", [])
    assert not asel._is_taxon("a committee having supervisory powers", ["commission"])
    assert not asel._is_taxon("a social class", ["working class"])


def test_anchor_rules_reject_mechanical_bad_examples():
    a = lambda lemma, pos="noun", common=True: {"lemma": lemma, "pos": pos, "wn_pos": "n", "synset": None,
                                                "gloss": "", "is_common": common}
    assert an._anchor_rule_verdict(None, a("tx"), a("us"), True) == "abbreviation or symbol"
    assert an._anchor_rule_verdict(None, a("u.s.a."), a("america"), True) == "abbreviation or symbol"
    assert an._anchor_rule_verdict(None, a("humans"), a("human"), True) == "same word in another form"
    assert an._anchor_rule_verdict(None, a("pub"), a("tavern", common=False), True) == "rare word"
    assert an._anchor_rule_verdict(None, a("walk", "verb"), a("go", "verb"), True) is None
    # vocab<->vocab anchors are rare on purpose -- no rarity rule for them
    assert an._anchor_rule_verdict(None, a("spinney", common=False), a("copse", common=False), False) is None


def _edge(cur, schema, a_id, b_id, rtype, family, pos_a="noun", pos_b="noun", anchor_status=None):
    cur.execute(
        f"""INSERT INTO {schema}.word_relation_edge
                (term_a_id, term_b_id, relation_type, relation_family, pos_a, pos_b, source,
                 verification_status, anchor_status)
            VALUES (%s,%s,%s,%s,%s,%s,'test','verified',%s)""",
        (a_id, b_id, rtype, family, pos_a, pos_b, anchor_status),
    )


@pg
def test_anchor_must_match_relation_type_and_pos_and_not_be_rejected():
    """The anchor has to test the same relation the question does: a
    holonym_member question never gets a holonym_part anchor, an adj->noun
    question never a noun->adj one, and a rejected anchor is never used."""
    schema = "cc_test_analogies_anchor_match"
    conn = db.connect(_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    db.apply_schema(conn, schema)
    try:
        with conn.cursor() as cur:
            def term(lemma, word_id=None, pos="n"):
                cur.execute(f"""INSERT INTO {schema}.wn_relation_term (word_id, lemma, wn_pos, synset_name, gloss, is_common)
                                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                            (word_id, lemma, pos, f"{lemma}.{pos}.01", f"{lemma} gloss", word_id is None))
                return cur.fetchone()[0]

            wid = _seed_word(cur, schema, "wolfling", "a young wolf")
            t_c, t_d = term("wolfling", wid), term("pack")
            _edge(cur, schema, t_c, t_d, "holonym_member", "part_of")
            # same family, wrong type; right type, wrong POS order; right type, rejected
            _edge(cur, schema, term("finger"), term("hand"), "holonym_part", "part_of")
            _edge(cur, schema, term("bee", pos="n"), term("swarming", pos="a"), "holonym_member", "part_of",
                  pos_b="adjective")
            _edge(cur, schema, term("tx"), term("us"), "holonym_member", "part_of", anchor_status="rejected")
            assert asel.select_analogy_edge(conn, schema, wid, exclude_ids=set()) is None

            _edge(cur, schema, term("sailor"), term("crew"), "holonym_member", "part_of", anchor_status="ok")
            assembly = asel.select_analogy_edge(conn, schema, wid, exclude_ids=set())
            assert assembly is not None
            assert (assembly.a_lemma, assembly.b_lemma) == ("sailor", "crew")
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


@pg
def test_traps_are_most_specific_first_not_alphabetical():
    """Regression: traps used to be sorted alphabetically and cut, so the
    near-root 'act'/'action' won on almost every is_a item. Now a lemma
    carried by many terms' fanout ranks below a specific one."""
    schema = "cc_test_analogies_trap_rank"
    conn = db.connect(_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    db.apply_schema(conn, schema)
    asel._generality_cache.clear()
    try:
        with conn.cursor() as cur:
            def term(lemma, word_id=None):
                cur.execute(f"""INSERT INTO {schema}.wn_relation_term (word_id, lemma, wn_pos, synset_name, gloss, is_common)
                                VALUES (%s, %s, 'n', %s, %s, %s) RETURNING id""",
                            (word_id, lemma, f"{lemma}.n.01", f"{lemma} gloss", word_id is None))
                return cur.fetchone()[0]

            def fan(term_id, lemma, pos="n"):
                cur.execute(f"""INSERT INTO {schema}.wn_relation_fanout (term_id, relation_type, target_lemma, target_pos)
                                VALUES (%s, 'hypernym', %s, %s)""", (term_id, lemma, pos))

            wid = {l: _seed_word(cur, schema, l, f"{l} def") for l in ("fetter", "shackle", "gauntlet", "vambrace")}
            t = {l: term(l, wid[l]) for l in wid}
            _edge(cur, schema, t["fetter"], t["shackle"], "hypernym", "is_a")
            _edge(cur, schema, t["gauntlet"], t["vambrace"], "hypernym", "is_a")
            for lemma in ("armor", "act", "action", "activity"):
                fan(t["gauntlet"], lemma)
            fan(t["gauntlet"], "armored", pos="a")  # wrong POS for a noun answer
            # 'act'/'action'/'activity' sit in 60 other terms' fanout -> generic
            for i in range(60):
                filler = term(f"filler{i}")
                for lemma in ("act", "action", "activity"):
                    fan(filler, lemma)

        assembly = asel.select_analogy_edge(conn, schema, wid["fetter"], exclude_ids=set())
        assert assembly is not None
        assert assembly.trap_lemmas == ["armor"]
    finally:
        asel._generality_cache.clear()
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


@pg
def test_contrast_options_come_from_cs_side_and_never_complete_the_family():
    """Wrong options are C's own neighbours through another relation (its
    sibling, its narrower kind) plus one of D's siblings; a neighbour that
    completes the tested family -- C's other hypernym on a definition-
    pattern kind_of item -- is a right answer and is excluded."""
    schema = "cc_test_analogies_contrast"
    conn = db.connect(_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    db.apply_schema(conn, schema)
    try:
        with conn.cursor() as cur:
            def term(lemma, word_id=None):
                cur.execute(f"""INSERT INTO {schema}.wn_relation_term (word_id, lemma, wn_pos, synset_name, gloss, is_common)
                                VALUES (%s, %s, 'n', %s, %s, %s) RETURNING id""",
                            (word_id, lemma, f"{lemma}.n.01", f"{lemma} gloss", word_id is None))
                return cur.fetchone()[0]

            def fan(term_id, rtype, lemma):
                cur.execute(f"""INSERT INTO {schema}.wn_relation_fanout (term_id, relation_type, target_lemma, target_pos)
                                VALUES (%s, %s, %s, 'n')""", (term_id, rtype, lemma))

            wid = _seed_word(cur, schema, "celandine", "a plant with yellow flowers")
            t_c, t_d = term("celandine", wid), term("herb")
            _edge(cur, schema, t_c, t_d, "definition_pattern_kind_of", "is_a")
            _edge(cur, schema, term("cure"), term("medication"), "hypernym", "is_a", anchor_status="ok")
            fan(t_c, "hypernym", "plant")                 # WordNet ancestor: a second right answer
            fan(t_c, "contrast_hypernym", "plant")
            fan(t_c, "sibling_of_hypernym_parent", "buttercup")
            fan(t_c, "contrast_hyponym", "lesser celandine")
            fan(t_d, "sibling_of_hypernym_parent", "shrub")

        a = asel.select_analogy_edge(conn, schema, wid, exclude_ids=set())
        assert a is not None
        assert "plant" in a.exclusion_lemmas
        got = dict(a.contrast_lemmas)
        assert "plant" not in got
        assert got["buttercup"] == "contrast_sibling"
        assert got["lesser celandine"] == "contrast_narrower"
        assert got["shrub"] == "d_sibling"
        assert a.anchor_edge_id is not None and a.tested_edge_id is not None
        assert a.relation_family == "is_a"
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")


def test_explanation_names_the_shared_relation():
    assert asel.explain("hypernym", "Cure", "medication", "CELANDINE", "herb") == (
        "In both pairs the first is a kind of the second: cure → medication, celandine → herb.")
    assert "opposites" in asel.explain("antonym", "hot", "cold", "wet", "dry")


def test_bank_check_rules():
    """solve must pick D with nothing else fitting; a majority of blind
    picks of D marks the item associative."""
    from concordance import analogy_bank as ab

    def item():
        return {"target_word_id": 1, "a_lemma": "cure", "b_lemma": "medication", "c_lemma": "celandine",
                "options": [{"label": x, "strategy": "s"} for x in ("buttercup", "shrub", "petal")]
                + [{"label": "herb", "strategy": "answer"}]}

    class FakeLLM:
        def __init__(self, solve_also=(), blind_right=True):
            self.solve_also, self.blind_right = solve_also, blind_right

        def create_chat_completion(self, messages, **_):
            import json as _json
            items = _json.loads(messages[1]["content"])
            out = []
            for it in items:
                herb = next(int(k) for k, v in it["options"].items() if v == "herb")
                if "a" in it:
                    also = [int(k) for k, v in it["options"].items() if v in self.solve_also]
                    out.append({"i": it["i"], "answer": herb, "also": also})
                else:
                    other = next(int(k) for k, v in it["options"].items() if v != "herb")
                    out.append({"i": it["i"], "answer": herb if self.blind_right else other})
            return {"choices": [{"message": {"content": _json.dumps(out)}}]}

    a, b, c = item(), item(), item()
    ab.check_items(FakeLLM(blind_right=False), [a], {1: "a plant"})
    ab.check_items(FakeLLM(blind_right=True), [b], {1: "a plant"})
    ab.check_items(FakeLLM(solve_also=("shrub",)), [c], {1: "a plant"})
    assert a["status"] == "ok" and a["blind_hits"] == 0
    assert b["status"] == "associative" and b["blind_hits"] == 2
    assert c["status"] == "failed" and "shrub" in c["check_note"]
    p = ab.payload_fields({**item(), "style": "one_hard_term", "relation_type": "hypernym", "relation_family": "is_a",
                           "tested_edge_id": 1, "anchor_edge_id": 2, "d_lemma": "herb"})
    assert next(o["label"] for o in p["options"] if o["key"] == p["correct_option_key"]) == "herb"


def test_option_pos_gate_rejects_words_mainly_another_pos():
    from concordance.distractors import pos_is_usual

    assert pos_is_usual("canoe", "noun")
    assert pos_is_usual("configure", "verb")
    assert not pos_is_usual("confection", "verb")   # mainly a noun
    assert not pos_is_usual("set", "adjective")     # mainly a verb
    assert pos_is_usual("xyzzyplugh", "noun")       # unknown to WordNet: nothing to judge by


def test_taxon_wholes_are_detected():
    from concordance.analogy_select import _synset_is_taxon, _whole_is_taxon

    assert _synset_is_taxon("vanellus.n.01")
    assert not _synset_is_taxon("lapwing.n.01")
    # madwort's member holonym 'alyssum' is genus Alyssum, not the plant
    assert _whole_is_taxon("alyssum.n.01", "alyssum")
    assert not _whole_is_taxon("trigger.n.01", "gun")


def test_seed_pairs_are_well_formed():
    from concordance.analogies import _RELATION_FAMILY
    from concordance.analogy_seed import seed_pairs

    pairs = seed_pairs()
    assert len(pairs) > 150
    for rtype, pos_a, pos_b, a, b in pairs:
        assert rtype in _RELATION_FAMILY
        assert a and b and a != b
    assert ("definition_pattern_purpose", "noun", "verb", "knife", "cut") in pairs
    assert ("attribute", "noun", "adjective", "weight", "heavy") in pairs
