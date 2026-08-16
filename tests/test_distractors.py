"""Distractor generation. Pure tests run always; DB-backed tests run only when a
throwaway Postgres is provided via CONCORDANCE_TEST_DB_URL (else skipped) --
same convention as test_db.py/test_auth.py."""

from __future__ import annotations

import os

import pytest

from concordance import db, distractors as dx


# --- pure helpers (no database) ---------------------------------------------

def test_distractor_config_defaults():
    cfg = dx.DistractorConfig()
    assert cfg.smart_vs_random_ratio == 0.7
    assert cfg.strategy_weights["antonym"] == 0.0
    assert set(cfg.strategy_weights) == {"orthographic", "semantic", "domain", "antonym"}


def test_distractor_result_defaults_not_degraded():
    result = dx.DistractorResult(candidates=[{"id": 1, "lemma": "x", "quiz_definition": None}])
    assert result.degraded is False


# --- DB-backed ---------------------------------------------------------------

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

_SCHEMA = "cc_test_distractors"


@pytest.fixture(scope="module")
def seeded():
    """A small, hand-controlled corpus: 2 POS, 2 USAS domains, and fasttext
    vectors placed at known cosine distances so the semantic-band strategy's
    behavior is verifiable rather than just "didn't crash"."""
    from pgvector.psycopg import register_vector

    conn = db.connect(_URL)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
    conn.commit()
    db.apply_schema(conn, _SCHEMA)

    def vec(*nonzero: tuple[int, float]) -> list[float]:
        v = [0.0] * 300
        for i, val in nonzero:
            v[i] = val
        return v

    # Every non-target word sits at difficulty=40 (comfortably inside a widened
    # band but outside a narrow one) -- see test_degraded_flag_..., which
    # requests a band tight enough that nothing qualifies pre-widen.
    words = {
        # lemma, pos, difficulty, synonyms, categories (USAS top-level codes), vector
        "target":       ("noun", 50.0, ["targetsyn"], ["B"], vec((0, 1.0))),
        "targetsyn":    ("noun", 40.0, [], ["B"], vec((0, 1.0))),          # excluded: is a synonym
        "toosclose":    ("noun", 40.0, [], ["B"], vec((0, 0.99), (1, 0.14107))),  # distance ~0.01
        # distance = 0.25 -- comfortably clear of _NEAR_SYNONYM_DISTANCE_FLOOR (0.20)
        # with margin for float rounding, not sitting right on the boundary.
        "nearmiss":     ("noun", 40.0, [], ["B"], vec((0, 0.75), (1, 0.661438))),
        "toofar":       ("noun", 40.0, [], [], vec((1, 1.0))),                    # distance = 1.0
        "samedomain":   ("noun", 40.0, [], ["B"], vec((2, 1.0))),          # domain match, far vector
        "otherdomain":  ("noun", 40.0, [], ["S"], vec((3, 1.0))),
        "lookalike":    ("noun", 40.0, [], [], vec((4, 1.0))),   # lemma "targetish" set below
        "verbword":     ("verb", 40.0, [], ["B"], vec((0, 1.0))),  # wrong POS -- must never appear
        "filler1":      ("noun", 40.0, [], [], vec((5, 1.0))),
        "filler2":      ("noun", 40.0, [], [], vec((6, 1.0))),
        "filler3":      ("noun", 50.0, [], [], vec((7, 1.0))),
    }

    ids = {}
    with conn.cursor() as cur:
        for lemma, (pos, diff, syns, cats, vector) in words.items():
            actual_lemma = "targetish" if lemma == "lookalike" else lemma
            cur.execute(
                f"""INSERT INTO {_SCHEMA}.word (lemma, definition, quiz_definition, part_of_speech, synonyms, active)
                    VALUES (%s, %s, %s, %s, %s, true) RETURNING id""",
                (actual_lemma, f"def of {actual_lemma}", f"quizdef of {actual_lemma}", pos, syns),
            )
            wid = cur.fetchone()[0]
            ids[lemma] = wid
            cur.execute(
                f"""INSERT INTO {_SCHEMA}.word_difficulty (word_id, quizzable, difficulty) VALUES (%s, true, %s)""",
                (wid, diff),
            )
            cur.execute(
                f"""INSERT INTO {_SCHEMA}.word_embedding (word_id, fasttext_vector, fasttext_model)
                    VALUES (%s, %s, 'test')""",
                (wid, vector),
            )
            for code in cats:
                cur.execute(
                    f"""INSERT INTO {_SCHEMA}.category (taxonomy, code, name)
                        VALUES ('usas', %s, %s) ON CONFLICT (taxonomy, code) DO NOTHING""",
                    (code, f"category {code}"),
                )
                cur.execute(f"SELECT id FROM {_SCHEMA}.category WHERE taxonomy='usas' AND code=%s", (code,))
                cat_id = cur.fetchone()[0]
                cur.execute(
                    f"INSERT INTO {_SCHEMA}.word_category (word_id, category_id, is_primary) VALUES (%s,%s,true)",
                    (wid, cat_id),
                )
    conn.commit()

    yield conn, ids

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_pos_is_never_violated(seeded):
    conn, ids = seeded
    cfg = dx.DistractorConfig()
    result = dx.select_mc_distractors(conn, _SCHEMA, ids["target"], "noun", cfg, count=8)
    assert ids["verbword"] not in {c["id"] for c in result.candidates}


@pg
def test_synonym_is_always_excluded(seeded):
    conn, ids = seeded
    cfg = dx.DistractorConfig(strategy_weights={"orthographic": 0, "semantic": 1.0, "domain": 0, "antonym": 0})
    result = dx.select_mc_distractors(conn, _SCHEMA, ids["target"], "noun", cfg, count=8)
    assert ids["targetsyn"] not in {c["id"] for c in result.candidates}


@pg
def test_semantic_band_excludes_too_close_and_too_far(seeded):
    conn, ids = seeded
    cfg = dx.DistractorConfig(smart_vs_random_ratio=1.0,
                               strategy_weights={"orthographic": 0, "semantic": 1.0, "domain": 0, "antonym": 0})
    result = dx.select_mc_distractors(conn, _SCHEMA, ids["target"], "noun", cfg, count=1)
    picked_ids = {c["id"] for c in result.candidates}
    assert ids["nearmiss"] in picked_ids
    assert ids["toosclose"] not in picked_ids
    assert ids["toofar"] not in picked_ids


@pg
def test_domain_strategy_matches_shared_category(seeded):
    conn, ids = seeded
    cfg = dx.DistractorConfig(smart_vs_random_ratio=1.0,
                               strategy_weights={"orthographic": 0, "semantic": 0, "domain": 1.0, "antonym": 0},
                               difficulty_min=None, difficulty_max=None)
    # count=1, satisfiable by the domain strategy alone (samedomain matches) --
    # keeps this a pure test of domain-matching, since asking for more would
    # let the random-fallback backfill legitimately pull in otherdomain too
    # (random doesn't care about domain, by design) and confound the assertion.
    exclude = {v for k, v in ids.items() if k not in ("target", "samedomain", "otherdomain")}
    result = dx.select_mc_distractors(conn, _SCHEMA, ids["target"], "noun", cfg, count=1,
                                       exclude_word_ids=exclude)
    picked_ids = {c["id"] for c in result.candidates}
    assert picked_ids == {ids["samedomain"]}


@pg
def test_near_synonym_floor_excludes_across_every_strategy_and_random_backfills(seeded):
    # Regression: live bug where "noctambule"/"noctambulist" (both meaning
    # "sleepwalker", both with an empty word.synonyms -- the only other
    # exclusion mechanism) appeared together as MC options. "toosclose" is
    # an embedding near-duplicate of "target" (distance ~0.01, same as that
    # real pair's ~0.15) but shares no lemma spelling and no synonyms entry
    # with it -- only a strategy with zero meaning-awareness of its own
    # (domain, here) can surface it, proving the floor is checked regardless
    # of *why* a strategy proposed the candidate, not just for "semantic".
    # Also confirms the rejected slot gets backfilled by random rather than
    # silently shrinking the result by one.
    conn, ids = seeded
    cfg = dx.DistractorConfig(smart_vs_random_ratio=1.0,
                               strategy_weights={"orthographic": 0, "semantic": 0, "domain": 1.0, "antonym": 0})
    exclude = {ids["targetsyn"], ids["nearmiss"], ids["samedomain"]}  # leaves "toosclose" as domain's only match
    result = dx.select_mc_distractors(conn, _SCHEMA, ids["target"], "noun", cfg, count=1,
                                       exclude_word_ids=exclude)
    picked_ids = {c["id"] for c in result.candidates}
    assert ids["toosclose"] not in picked_ids
    assert len(result.candidates) == 1
    assert result.candidates[0]["strategy"] == "random"


@pg
def test_orthographic_strategy_matches_similar_lemma(seeded):
    conn, ids = seeded
    cfg = dx.DistractorConfig(smart_vs_random_ratio=1.0,
                               strategy_weights={"orthographic": 1.0, "semantic": 0, "domain": 0, "antonym": 0})
    result = dx.select_mc_distractors(conn, _SCHEMA, ids["target"], "noun", cfg, count=1)
    assert {c["id"] for c in result.candidates} == {ids["lookalike"]}


@pg
def test_random_fallback_fills_shortfall_without_erroring(seeded):
    conn, ids = seeded
    # No embedding-based/domain-based signal can possibly satisfy this (100%
    # semantic weight, but ask for far more than the near-miss band has) --
    # random must make up the deficit rather than returning short or raising.
    cfg = dx.DistractorConfig(smart_vs_random_ratio=1.0,
                               strategy_weights={"orthographic": 0, "semantic": 1.0, "domain": 0, "antonym": 0})
    result = dx.select_mc_distractors(conn, _SCHEMA, ids["target"], "noun", cfg, count=6)
    assert len(result.candidates) == 6
    assert any(c["strategy"] == "random" for c in result.candidates)


@pg
def test_degraded_flag_set_on_pathologically_narrow_difficulty_band(seeded):
    conn, ids = seeded
    cfg = dx.DistractorConfig(difficulty_min=49.999, difficulty_max=50.001)
    result = dx.select_mc_distractors(conn, _SCHEMA, ids["target"], "noun", cfg, count=6)
    assert result.degraded is True
    assert len(result.candidates) == 6  # still filled, via the widened last resort


@pg
def test_require_quiz_definition_filters_candidates_without_one(seeded):
    conn, ids = seeded
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {_SCHEMA}.word SET quiz_definition = NULL WHERE id = %s", (ids["nearmiss"],))
    conn.commit()
    try:
        cfg = dx.DistractorConfig(smart_vs_random_ratio=1.0,
                                   strategy_weights={"orthographic": 0, "semantic": 1.0, "domain": 0, "antonym": 0})
        result = dx.select_mc_distractors(conn, _SCHEMA, ids["target"], "noun", cfg, count=1,
                                           require_quiz_definition=True)
        assert ids["nearmiss"] not in {c["id"] for c in result.candidates}
    finally:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE {_SCHEMA}.word SET quiz_definition = 'quizdef of nearmiss' WHERE id = %s",
                        (ids["nearmiss"],))
        conn.commit()


@pg
def test_select_tf_foil_returns_one_word_with_quiz_definition(seeded):
    conn, ids = seeded
    cfg = dx.DistractorConfig()
    foil = dx.select_tf_foil(conn, _SCHEMA, ids["target"], "noun", cfg)
    assert foil is not None
    assert foil["id"] != ids["target"]
    assert foil["quiz_definition"]


@pg
def test_select_matching_set_returns_set_size_minus_one_words(seeded):
    conn, ids = seeded
    cfg = dx.DistractorConfig()
    result = dx.select_matching_set(conn, _SCHEMA, ids["target"], "noun", cfg, set_size=4)
    assert len(result.candidates) == 3
    assert all(c["quiz_definition"] for c in result.candidates)


# --- analogy distractors: same near-synonym floor, own code path ------------

@pg
def test_analogy_distractors_apply_near_synonym_floor_to_trap_candidates():
    # d_word_id threads the same _NEAR_SYNONYM_DISTANCE_FLOOR check into
    # select_analogy_distractors -- a trap candidate (A/B reused in D's slot,
    # normally deliberately PREFERRED, see module docstring) that happens to
    # also be a near-synonym of D is just as bad a distractor here as it
    # would be for mc/true_false/matching.
    from pgvector.psycopg import register_vector

    schema = "cc_test_analogy_near_synonym"
    conn = db.connect(_URL)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    def vec(*nonzero):
        v = [0.0] * 300
        for i, val in nonzero:
            v[i] = val
        return v

    ids = {}
    with conn.cursor() as cur:
        for lemma, vector in [
            ("dword", vec((0, 1.0))),
            ("closeword", vec((0, 0.99), (1, 0.14107))),  # distance ~0.01 from dword
            ("farword", vec((1, 1.0))),                    # distance 1.0 from dword
        ]:
            cur.execute(
                f"""INSERT INTO {schema}.word (lemma, definition, quiz_definition, part_of_speech, active)
                    VALUES (%s,%s,%s,'noun',true) RETURNING id""",
                (lemma, f"def {lemma}", f"quizdef {lemma}"))
            wid = cur.fetchone()[0]
            ids[lemma] = wid
            cur.execute(f"INSERT INTO {schema}.word_difficulty (word_id, quizzable, difficulty) VALUES (%s,true,50.0)",
                        (wid,))
            cur.execute(f"INSERT INTO {schema}.word_embedding (word_id, fasttext_vector, fasttext_model) "
                        f"VALUES (%s,%s,'test')", (wid, vector))
    conn.commit()

    result = dx.select_analogy_distractors(
        conn, schema, "vocab_only", "noun", "a_lemma", "b_lemma",
        c_word_id=ids["farword"], d_term_id=0, exclusion_lemmas=set(),
        trap_lemmas=["closeword", "farword"], count=1,
        exclude_word_ids={ids["dword"]}, d_word_id=ids["dword"],
    )
    picked_lemmas = {c["lemma"] for c in result.candidates}
    assert "closeword" not in picked_lemmas

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_analogy_distractors_skip_floor_check_when_d_has_no_word_row():
    # one_hard_term style's D is frequently an ordinary WordNet term with no
    # word row at all (d_word_id=None, see the function's own docstring) --
    # must degrade gracefully (no crash, no filtering attempted) rather than
    # erroring on a missing embedding to compare against.
    schema = "cc_test_analogy_no_d_word"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {schema}.word (lemma, definition, quiz_definition, part_of_speech, active)
                VALUES ('trapword','def','quizdef','noun',true) RETURNING id""")
        trap_id = cur.fetchone()[0]
        cur.execute(f"INSERT INTO {schema}.word_difficulty (word_id, quizzable, difficulty) VALUES (%s,true,50.0)",
                    (trap_id,))
        cur.execute(
            f"""INSERT INTO {schema}.wn_relation_term (word_id, lemma, wn_pos, is_common)
                VALUES (NULL, 'ordinaryword', 'n', true)""")
    conn.commit()

    result = dx.select_analogy_distractors(
        conn, schema, "one_hard_term", "noun", "a_lemma", "b_lemma",
        c_word_id=None, d_term_id=0, exclusion_lemmas=set(),
        trap_lemmas=["trapword"], count=1, d_word_id=None,
    )
    assert len(result.candidates) == 1  # no crash, trap candidate accepted normally

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()
