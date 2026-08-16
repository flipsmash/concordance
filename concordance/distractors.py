"""Distractor generation for quiz questions (§ quizzing).

Given a target word, produces plausible-but-wrong option words for multiple-choice,
a foil word for true/false, or a set of companion words for matching -- all POS-matched
(non-negotiable) and drawn from a weighted blend of strategies:

  orthographic  looks like the target (pg_trgm lemma similarity) -- only meaningful
                when the *word* is what a quiz-taker chooses among, not its definition
  semantic      near-miss embedding proximity: close enough to be a plausible mix-up,
                far enough not to be a true synonym (a distance *band*, not nearest-only)
  domain        shares a top-level USAS category with the target
  antonym       stubbed -- no antonym data exists anywhere in this pipeline yet; the
                weight key is reserved so a future data source is a drop-in, not a
                config-shape migration
  random        any other eligible word -- also the universal fallback for every
                other strategy's shortfall

Generated live at quiz-start time (not pre-cached): selection depends on per-session
parameters (difficulty range, filters, ratios) that don't cleanly cache across configs,
and this is the same query shape webapp/backend/main.py's word_neighbors already proves
fast enough per-word at this corpus size.

Fallback rule: POS and the difficulty band are never relaxed to fill a shortfall -- a
strategy's deficit spills to the next-weighted strategy, then to random. Only if random
itself can't fill the count under POS+difficulty (a pathologically narrow config) does
the difficulty band widen symmetrically as a last resort, which the caller can see via
DistractorResult.degraded rather than a silent shortfall.

A target's own synonyms are always excluded, from every strategy -- a distractor that's
actually a valid synonym of the correct answer isn't wrong, it's a second correct answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# Cosine-distance band for the "semantic near-miss" strategy: too close (< _SEMANTIC_BAND_MIN)
# risks being a near-synonym even after the explicit synonym exclusion below; too far
# (> _SEMANTIC_BAND_MAX) isn't a plausible mix-up anymore. Starting heuristic, expected to
# need empirical tuning once real quiz data exists -- not an architectural commitment.
_SEMANTIC_BAND_MIN = 0.15
_SEMANTIC_BAND_MAX = 0.45

# Minimum pg_trgm similarity for a lemma to count as an "orthographic lookalike" at all,
# rather than just a coincidental trigram overlap.
_ORTHOGRAPHIC_MIN_SIMILARITY = 0.3

# Hard exclusion floor applied to EVERY strategy's candidates (not just "semantic"),
# distinct from and stricter than _SEMANTIC_BAND_MIN above. word.synonyms (the OTHER
# synonym-exclusion mechanism -- see module docstring) is empty for many rare/obscure
# words, so a genuine near-synonym can still slip through: confirmed live, "noctambule"/
# "noctambulist" (both meaning "sleepwalker", both with an empty synonyms column) --
# orthographic trigram similarity 0.6 (well past _ORTHOGRAPHIC_MIN_SIMILARITY, which has
# no meaning-awareness of its own at all) AND definition-embedding distance 0.151 (a hair
# inside the OLD 0.15 semantic near-miss floor). This floor is checked regardless of which
# strategy proposed the candidate, precisely because orthographic similarity proved just as
# capable of surfacing a true synonym as the semantic band was.
_NEAR_SYNONYM_DISTANCE_FLOOR = 0.20

# How far a last-resort difficulty-band widen reaches, symmetrically, when even random
# can't fill the requested count under the original band.
_DEGRADED_WIDEN_POINTS = 15.0

_SIGNAL_COLUMNS = ("definition_vector", "fasttext_vector")


@dataclass
class DistractorConfig:
    difficulty_min: float | None = None
    difficulty_max: float | None = None
    # fraction of the requested count drawn from smart strategies vs. random
    smart_vs_random_ratio: float = 0.7
    # relative weights among the smart strategies; a zero/absent key contributes nothing
    # (e.g. the caller zeroes 'orthographic' under the word_to_definition direction, where
    # a lookalike *word* is invisible since only definitions are shown as options)
    strategy_weights: dict[str, float] = field(
        default_factory=lambda: {"orthographic": 1 / 3, "semantic": 1 / 3, "domain": 1 / 3, "antonym": 0.0}
    )


@dataclass
class DistractorResult:
    candidates: list[dict]  # each: {id, lemma, quiz_definition, strategy}
    degraded: bool = False  # True if the difficulty band had to widen to fill the count,
                             # or the count still couldn't be fully filled


def _target_info(conn, schema: str, word_id: int) -> tuple[str, list[str]]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT lemma, synonyms FROM {schema}.word WHERE id = %s", (word_id,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"word {word_id} not found")
    return row[0], list(row[1] or [])


def _embedding_signal(conn, schema: str, word_id: int) -> tuple[object | None, str | None]:
    """(vector, column_name) for word_id's definition_vector, or its
    fasttext_vector if that's missing, or (None, None) if neither exists --
    same column-priority _semantic_band_candidates already uses."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT definition_vector, fasttext_vector FROM {schema}.word_embedding WHERE word_id = %s",
                    (word_id,))
        row = cur.fetchone()
    if row is None:
        return None, None
    for i, col in enumerate(_SIGNAL_COLUMNS):
        if row[i] is not None:
            return row[i], col
    return None, None


def _too_similar_ids(conn, schema: str, target_word_id: int, candidate_ids: list,
                      floor: float = _NEAR_SYNONYM_DISTANCE_FLOOR) -> set[int]:
    """Candidate word ids whose meaning is too close to the target to be a
    fair distractor -- checked regardless of which strategy proposed them
    (see _NEAR_SYNONYM_DISTANCE_FLOOR's own comment). A candidate that
    doesn't share the target's embedding column (missing data on either
    side) can't be evaluated and is never included here -- consistent with
    this module's existing stance elsewhere of not blocking on missing
    signal (an under-strict check, never an over-strict one)."""
    ids = [c for c in candidate_ids if c is not None]
    if not ids:
        return set()
    target_vec, col = _embedding_signal(conn, schema, target_word_id)
    if target_vec is None:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT word_id FROM {schema}.word_embedding
                WHERE word_id = ANY(%s) AND {col} IS NOT NULL AND ({col} <=> %s::vector) < %s""",
            (ids, target_vec, floor),
        )
        return {r[0] for r in cur.fetchall()}


def _base_filters(pos: str, cfg: DistractorConfig, exclude_ids: set[int], exclude_lemmas: list[str],
                   require_quiz_definition: bool) -> tuple[list[str], list]:
    filters = ["w.active", "wd.quizzable = true", "w.part_of_speech = %s", "NOT (w.id = ANY(%s))"]
    params: list = [pos, list(exclude_ids)]
    if exclude_lemmas:
        filters.append("NOT (lower(w.lemma) = ANY(%s))")
        params.append([lem.lower() for lem in exclude_lemmas])
    if cfg.difficulty_min is not None:
        filters.append("wd.difficulty >= %s")
        params.append(cfg.difficulty_min)
    if cfg.difficulty_max is not None:
        filters.append("wd.difficulty <= %s")
        params.append(cfg.difficulty_max)
    if require_quiz_definition:
        filters.append("w.quiz_definition IS NOT NULL")
    return filters, params


def _orthographic_candidates(conn, schema, target_lemma, pos, cfg, exclude_ids, exclude_lemmas,
                              require_quiz_definition, limit) -> list[dict]:
    filters, params = _base_filters(pos, cfg, exclude_ids, exclude_lemmas, require_quiz_definition)
    filters.append("similarity(w.lemma, %s) > %s")
    params.extend([target_lemma, _ORTHOGRAPHIC_MIN_SIMILARITY])
    where = " AND ".join(filters)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT w.id, w.lemma, w.quiz_definition
                FROM {schema}.word w
                JOIN {schema}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
                ORDER BY similarity(w.lemma, %s) DESC
                LIMIT %s""",
            (*params, target_lemma, limit),
        )
        rows = cur.fetchall()
    return [{"id": r[0], "lemma": r[1], "quiz_definition": r[2]} for r in rows]


def _semantic_band_candidates(conn, schema, target_word_id, pos, cfg, exclude_ids, exclude_lemmas,
                               require_quiz_definition, limit) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT definition_vector, fasttext_vector FROM {schema}.word_embedding WHERE word_id = %s",
            (target_word_id,),
        )
        row = cur.fetchone()
    if row is None:
        return []
    vec_col = None
    for i, col in enumerate(_SIGNAL_COLUMNS):
        if row[i] is not None:
            vec_col = col
            break
    if vec_col is None:
        return []

    filters, params = _base_filters(pos, cfg, exclude_ids, exclude_lemmas, require_quiz_definition)
    filters.append(f"e.{vec_col} IS NOT NULL")
    filters.append(f"(e.{vec_col} <=> (SELECT {vec_col} FROM {schema}.word_embedding WHERE word_id = %s)) BETWEEN %s AND %s")
    params.extend([target_word_id, _SEMANTIC_BAND_MIN, _SEMANTIC_BAND_MAX])
    where = " AND ".join(filters)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT w.id, w.lemma, w.quiz_definition,
                       e.{vec_col} <=> (SELECT {vec_col} FROM {schema}.word_embedding WHERE word_id = %s) AS distance
                FROM {schema}.word_embedding e
                JOIN {schema}.word w ON w.id = e.word_id
                JOIN {schema}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
                ORDER BY distance
                LIMIT %s""",
            (target_word_id, *params, limit),
        )
        rows = cur.fetchall()
    return [{"id": r[0], "lemma": r[1], "quiz_definition": r[2]} for r in rows]


def _domain_candidates(conn, schema, target_word_id, pos, cfg, exclude_ids, exclude_lemmas,
                        require_quiz_definition, limit) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT DISTINCT left(c.code, 1) FROM {schema}.word_category wc
                JOIN {schema}.category c ON c.id = wc.category_id
                WHERE wc.word_id = %s""",
            (target_word_id,),
        )
        target_fields = [r[0] for r in cur.fetchall()]
    if not target_fields:
        return []

    filters, params = _base_filters(pos, cfg, exclude_ids, exclude_lemmas, require_quiz_definition)
    filters.append(
        f"""EXISTS (SELECT 1 FROM {schema}.word_category wc2
                    JOIN {schema}.category c2 ON c2.id = wc2.category_id
                    WHERE wc2.word_id = w.id AND left(c2.code, 1) = ANY(%s))"""
    )
    params.append(target_fields)
    where = " AND ".join(filters)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT w.id, w.lemma, w.quiz_definition
                FROM {schema}.word w
                JOIN {schema}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
                ORDER BY random()
                LIMIT %s""",
            (*params, limit),
        )
        rows = cur.fetchall()
    return [{"id": r[0], "lemma": r[1], "quiz_definition": r[2]} for r in rows]


def _antonym_candidates(conn, schema, target_word_id, pos, cfg, exclude_ids, exclude_lemmas,
                         require_quiz_definition, limit) -> list[dict]:
    # No antonym data source exists anywhere in this pipeline (confirmed via full-repo
    # grep during planning). Reserved so a future data source is a config drop-in.
    return []


def _random_candidates(conn, schema, pos, cfg, exclude_ids, exclude_lemmas,
                        require_quiz_definition, limit) -> list[dict]:
    filters, params = _base_filters(pos, cfg, exclude_ids, exclude_lemmas, require_quiz_definition)
    where = " AND ".join(filters)
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT w.id, w.lemma, w.quiz_definition
                FROM {schema}.word w
                JOIN {schema}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
                ORDER BY random()
                LIMIT %s""",
            (*params, limit),
        )
        rows = cur.fetchall()
    return [{"id": r[0], "lemma": r[1], "quiz_definition": r[2]} for r in rows]


_STRATEGY_FUNCS = {
    "orthographic": lambda conn, schema, wid, lemma, pos, cfg, ex_ids, ex_lem, rqd, limit:
        _orthographic_candidates(conn, schema, lemma, pos, cfg, ex_ids, ex_lem, rqd, limit),
    "semantic": lambda conn, schema, wid, lemma, pos, cfg, ex_ids, ex_lem, rqd, limit:
        _semantic_band_candidates(conn, schema, wid, pos, cfg, ex_ids, ex_lem, rqd, limit),
    "domain": lambda conn, schema, wid, lemma, pos, cfg, ex_ids, ex_lem, rqd, limit:
        _domain_candidates(conn, schema, wid, pos, cfg, ex_ids, ex_lem, rqd, limit),
    "antonym": lambda conn, schema, wid, lemma, pos, cfg, ex_ids, ex_lem, rqd, limit:
        _antonym_candidates(conn, schema, wid, pos, cfg, ex_ids, ex_lem, rqd, limit),
}


def _select_candidates(conn, schema: str, target_word_id: int, pos: str, cfg: DistractorConfig,
                        count: int, exclude_word_ids: set[int],
                        require_quiz_definition: bool) -> DistractorResult:
    target_lemma, synonyms = _target_info(conn, schema, target_word_id)
    exclude_ids = set(exclude_word_ids) | {target_word_id}
    picked: list[dict] = []

    def _accept(got: list[dict], strategy: str) -> int:
        # Checked for every strategy's output, not just "semantic" -- see
        # _NEAR_SYNONYM_DISTANCE_FLOOR's own comment (orthographic and even
        # random candidates can turn out to be true synonyms of the target
        # too, and word.synonyms -- the OTHER exclusion mechanism -- is
        # empty for many rare/obscure words). Returns the count actually
        # accepted (not len(got)) -- callers must use this, not len(got),
        # to track remaining need, or a rejected candidate's slot silently
        # never gets filled instead of spilling to the next strategy/random
        # per this function's own documented fallback rule.
        too_close = _too_similar_ids(conn, schema, target_word_id, [c["id"] for c in got])
        accepted = 0
        for c in got:
            if c["id"] in exclude_ids or c["id"] in too_close:
                exclude_ids.add(c["id"])
                continue
            c["strategy"] = strategy
            picked.append(c)
            exclude_ids.add(c["id"])
            accepted += 1
        return accepted

    smart_count = round(count * cfg.smart_vs_random_ratio)
    active_weights = {k: v for k, v in cfg.strategy_weights.items() if v and v > 0 and k in _STRATEGY_FUNCS}
    total_w = sum(active_weights.values()) or 1.0
    remaining_smart = smart_count

    for strat, weight in sorted(active_weights.items(), key=lambda kv: -kv[1]):
        if remaining_smart <= 0:
            break
        alloc = min(round(smart_count * (weight / total_w)), remaining_smart)
        if alloc <= 0:
            continue
        got = _STRATEGY_FUNCS[strat](conn, schema, target_word_id, target_lemma, pos, cfg,
                                      exclude_ids, synonyms, require_quiz_definition, alloc)
        remaining_smart -= _accept(got, strat)

    # Retried (not just one shot): _random_candidates' own LIMIT is computed
    # BEFORE the too-similar filter runs, so a batch that happens to include
    # a near-synonym comes back short after _accept rejects it -- each
    # retry's exclude_ids already covers everything the previous attempt
    # saw (accepted or rejected), so it always asks for candidates not yet
    # tried, converging to either `count` or a genuinely exhausted pool.
    # Bounded attempts as a defensive cap only; the empty-`got` break is
    # what actually guarantees termination.
    random_count = (count - len(picked))
    degraded = False
    for _ in range(5):
        if random_count <= 0:
            break
        got = _random_candidates(conn, schema, pos, cfg, exclude_ids, synonyms,
                                  require_quiz_definition, random_count)
        if not got:
            break
        random_count -= _accept(got, "random")

    if random_count > 0 and (cfg.difficulty_min is not None or cfg.difficulty_max is not None):
        widened = replace(
            cfg,
            difficulty_min=None if cfg.difficulty_min is None else max(0.0, cfg.difficulty_min - _DEGRADED_WIDEN_POINTS),
            difficulty_max=None if cfg.difficulty_max is None else min(100.0, cfg.difficulty_max + _DEGRADED_WIDEN_POINTS),
        )
        for _ in range(5):
            if random_count <= 0:
                break
            got = _random_candidates(conn, schema, pos, widened, exclude_ids, synonyms,
                                      require_quiz_definition, random_count)
            if not got:
                break
            random_count -= _accept(got, "random")
        degraded = True

    return DistractorResult(candidates=picked, degraded=degraded or random_count > 0)


def select_mc_distractors(conn, schema: str, target_word_id: int, pos: str, cfg: DistractorConfig,
                           count: int, exclude_word_ids: set[int] | None = None,
                           require_quiz_definition: bool = False) -> DistractorResult:
    """Up to `count` distractor words for a multiple-choice question about
    target_word_id. `require_quiz_definition=True` under the word_to_definition
    direction, where each option's quiz_definition (not just its lemma) is shown."""
    return _select_candidates(conn, schema, target_word_id, pos, cfg, count,
                               exclude_word_ids or set(), require_quiz_definition)


def select_tf_foil(conn, schema: str, target_word_id: int, pos: str,
                    cfg: DistractorConfig, exclude_word_ids: set[int] | None = None) -> dict | None:
    """One word whose quiz_definition will be shown as the false statement for a
    true/false question about target_word_id. None only if the corpus genuinely
    has no eligible word left (pathological config)."""
    result = _select_candidates(conn, schema, target_word_id, pos, cfg, 1,
                                 exclude_word_ids or set(), require_quiz_definition=True)
    return result.candidates[0] if result.candidates else None


def select_matching_set(conn, schema: str, seed_word_id: int, pos: str, cfg: DistractorConfig,
                         set_size: int, exclude_word_ids: set[int] | None = None) -> DistractorResult:
    """set_size - 1 additional real words (with quiz_definition) to pair with
    seed_word_id into one matching block. The strategies apply to *which words
    belong in the set* here, not to synthetic option generation -- the wrong
    pairings in the rendered matching UI are these words' own real definitions."""
    return _select_candidates(conn, schema, seed_word_id, pos, cfg, set_size - 1,
                               exclude_word_ids or set(), require_quiz_definition=True)


# --- analogy distractors (§ analogies) -----------------------------------------
#
# For an A:B::C:? item, wrong options must additionally exclude every word that
# satisfies the SAME relation to C (see concordance/analogy_select.py's
# exclusion_lemmas -- the full, non-vocab-restricted WordNet target set,
# transitive-closure where applicable, precomputed by concordance/analogies.py)
# plus D's own synonym set -- this is the synonym-exclusion rule above, extended
# to cover "second valid completion of the same relation," not just literal
# synonymy. Conversely, a distractor that completes the SAME relation using A or
# B instead of C -- trap_lemmas below -- is the authentic MAT "wrong term, right
# relation" trap and is deliberately PREFERRED, not excluded.

def _resolve_word_by_lemma_pos(cur, schema: str, lemma: str, pos: str) -> dict | None:
    cur.execute(
        f"""SELECT w.id, w.lemma, w.quiz_definition FROM {schema}.word w
            JOIN {schema}.word_difficulty wd ON wd.word_id = w.id
            WHERE w.active AND wd.quizzable = true AND lower(w.lemma) = %s AND w.part_of_speech = %s
            LIMIT 1""",
        (lemma.lower(), pos),
    )
    row = cur.fetchone()
    return {"id": row[0], "lemma": row[1], "quiz_definition": row[2]} if row else None


def _trap_candidates(cur, schema: str, trap_lemmas: list[str], d_pos: str,
                      exclude_ids: set[int], limit: int) -> list[dict]:
    out = []
    for lemma in trap_lemmas:
        if len(out) >= limit:
            break
        word = _resolve_word_by_lemma_pos(cur, schema, lemma, d_pos)
        if word:
            if word["id"] in exclude_ids:
                continue
            out.append({"id": word["id"], "lemma": word["lemma"], "quiz_definition": word["quiz_definition"],
                        "strategy": "analogy_trap"})
        else:
            out.append({"id": None, "lemma": lemma, "quiz_definition": None, "strategy": "analogy_trap"})
    return out


def _embedding_offset_candidates(cur, schema: str, a_lemma: str, b_lemma: str, c_word_id: int, d_pos: str,
                                  exclude_lemmas: set[str], exclude_ids: set[int], limit: int) -> list[dict]:
    """vocab_only style only: vec(B) - vec(A) + vec(C), nearest vocab words
    matching d_pos -- computed entirely in SQL (pgvector supports vector
    arithmetic operators), same word_embedding.definition_vector HNSW index
    _semantic_band_candidates already reads."""
    cur.execute(
        f"""SELECT
                (SELECT e.definition_vector FROM {schema}.word_embedding e
                     JOIN {schema}.word w ON w.id = e.word_id WHERE lower(w.lemma) = %s LIMIT 1),
                (SELECT e.definition_vector FROM {schema}.word_embedding e
                     JOIN {schema}.word w ON w.id = e.word_id WHERE lower(w.lemma) = %s LIMIT 1),
                (SELECT e.definition_vector FROM {schema}.word_embedding e WHERE e.word_id = %s)""",
        (a_lemma.lower(), b_lemma.lower(), c_word_id),
    )
    a_vec, b_vec, c_vec = cur.fetchone()
    if a_vec is None or b_vec is None or c_vec is None:
        return []
    cur.execute(
        f"""SELECT w.id, w.lemma, w.quiz_definition
            FROM {schema}.word_embedding e
            JOIN {schema}.word w ON w.id = e.word_id
            JOIN {schema}.word_difficulty wd ON wd.word_id = w.id
            WHERE w.active AND wd.quizzable = true AND w.part_of_speech = %s
              AND NOT (w.id = ANY(%s)) AND e.definition_vector IS NOT NULL
            ORDER BY e.definition_vector <=> ((%s::vector) - (%s::vector) + (%s::vector))
            LIMIT %s""",
        (d_pos, list(exclude_ids), b_vec, a_vec, c_vec, limit * 3),
    )
    out = []
    for wid, lemma, qdef in cur.fetchall():
        if lemma.lower() in exclude_lemmas:
            continue
        out.append({"id": wid, "lemma": lemma, "quiz_definition": qdef, "strategy": "embedding_offset"})
        if len(out) >= limit:
            break
    return out


def _sibling_fanout_candidates(cur, schema: str, d_term_id: int, d_pos: str,
                                exclude_lemmas: set[str], exclude_ids: set[int], limit: int) -> list[dict]:
    """one_hard_term style only: co-hyponyms of D under D's own hypernym
    parent, precomputed by analogies.compute_sibling_fanout -- empty for any
    D whose verified edge wasn't in the is_a family (compute_sibling_fanout
    is only ever called for those), in which case this simply falls through
    to the random ordinary-term fallback."""
    cur.execute(
        f"""SELECT target_lemma FROM {schema}.wn_relation_fanout
            WHERE term_id = %s AND relation_type = 'sibling_of_hypernym_parent'
            ORDER BY random() LIMIT %s""",
        (d_term_id, limit * 3),
    )
    out = []
    for (lemma,) in cur.fetchall():
        if lemma.lower() in exclude_lemmas:
            continue
        word = _resolve_word_by_lemma_pos(cur, schema, lemma, d_pos)
        if word and word["id"] in exclude_ids:
            continue
        out.append({"id": word["id"] if word else None, "lemma": lemma,
                    "quiz_definition": word["quiz_definition"] if word else None, "strategy": "sibling_fanout"})
        if len(out) >= limit:
            break
    return out


def _random_ordinary_candidates(cur, schema: str, d_pos: str, exclude_lemmas: set[str], limit: int) -> list[dict]:
    """Last-resort fill for the one_hard_term style: a random common
    (is_common) ordinary term of the right POS, not tied to any word row."""
    from .model import wordnet_pos

    wn_pos = wordnet_pos(d_pos) or "n"
    cur.execute(
        f"""SELECT lemma FROM {schema}.wn_relation_term
            WHERE word_id IS NULL AND wn_pos = %s AND is_common
            ORDER BY random() LIMIT %s""",
        (wn_pos, limit * 3),
    )
    out = []
    for (lemma,) in cur.fetchall():
        if lemma.lower() in exclude_lemmas:
            continue
        out.append({"id": None, "lemma": lemma, "quiz_definition": None, "strategy": "random_ordinary"})
        if len(out) >= limit:
            break
    return out


def select_analogy_distractors(conn, schema: str, style: str, d_pos: str, a_lemma: str, b_lemma: str,
                                c_word_id: int, d_term_id: int, exclusion_lemmas: set[str],
                                trap_lemmas: list[str], count: int,
                                exclude_word_ids: set[int] | None = None,
                                d_word_id: int | None = None) -> DistractorResult:
    """`count` wrong D-options for an analogy item. `d_pos` is the canonical
    POS (noun/verb/adjective/adverb) of the correct answer D, used to
    POS-match every strategy. `c_word_id` feeds the vocab_only style's
    embedding-offset heuristic; `d_term_id` (D's wn_relation_term id, NOT its
    word id -- D is frequently an ordinary term with no word row at all in
    the one_hard_term style) feeds that style's precomputed sibling-fanout
    lookup. `d_word_id` is D's own word id when it has one (None for an
    ordinary WordNet term with no word row, same case as above) -- feeds the
    near-synonym distance floor below, skipped entirely when None since
    there's no embedding to check against. Every candidate, regardless of
    source, is checked against `exclusion_lemmas` (trap and plausibility
    strategies are equally capable of accidentally proposing a second right
    answer -- see module docstring above) AND this same distance floor
    _select_candidates uses for mc/true_false/matching (see
    _NEAR_SYNONYM_DISTANCE_FLOOR): a trap/plausibility candidate that
    happens to mean the same thing as D is just as bad a distractor here as
    it would be anywhere else."""
    exclude_ids = set(exclude_word_ids or set())
    exclusion_lemmas = {l.lower() for l in exclusion_lemmas}
    picked: list[dict] = []

    def _too_close(cands: list[dict]) -> set[int]:
        if d_word_id is None:
            return set()
        return _too_similar_ids(conn, schema, d_word_id, [c["id"] for c in cands])

    with conn.cursor() as cur:
        trap_batch = _trap_candidates(cur, schema, trap_lemmas, d_pos, exclude_ids, count)
        too_close = _too_close(trap_batch)
        for c in trap_batch:
            if c["lemma"].lower() in exclusion_lemmas or c["lemma"].lower() in {p["lemma"].lower() for p in picked}:
                continue
            if c["id"] in too_close:
                continue
            picked.append(c)
            if c["id"]:
                exclude_ids.add(c["id"])
            if len(picked) >= count:
                break

        remaining = count - len(picked)
        degraded = False
        if remaining > 0:
            if style == "vocab_only":
                more = _embedding_offset_candidates(cur, schema, a_lemma, b_lemma, c_word_id, d_pos,
                                                     exclusion_lemmas, exclude_ids, remaining)
            else:
                more = _sibling_fanout_candidates(cur, schema, d_term_id, d_pos, exclusion_lemmas,
                                                   exclude_ids, remaining)
            too_close = _too_close(more)
            for c in more:
                if c["lemma"].lower() in {p["lemma"].lower() for p in picked}:
                    continue
                if c["id"] in too_close:
                    continue
                picked.append(c)
                if c["id"]:
                    exclude_ids.add(c["id"])
            remaining = count - len(picked)

        if remaining > 0:
            if style == "vocab_only":
                cfg = DistractorConfig(strategy_weights={"orthographic": 0, "semantic": 0, "domain": 0, "antonym": 0})
                more = _random_candidates(conn, schema, d_pos, cfg, exclude_ids, [], False, remaining)
                too_close = _too_close(more)
                for c in more:
                    c["strategy"] = "random"
                    if c["lemma"].lower() in exclusion_lemmas or c["lemma"].lower() in {p["lemma"].lower() for p in picked}:
                        continue
                    if c["id"] in too_close:
                        continue
                    picked.append(c)
                    exclude_ids.add(c["id"])
            else:
                more = _random_ordinary_candidates(cur, schema, d_pos, exclusion_lemmas, remaining)
                too_close = _too_close(more)
                for c in more:
                    if c["lemma"].lower() in {p["lemma"].lower() for p in picked}:
                        continue
                    if c["id"] in too_close:
                        continue
                    picked.append(c)
            degraded = degraded or (len(picked) < count)

    return DistractorResult(candidates=picked[:count], degraded=degraded or len(picked) < count)
